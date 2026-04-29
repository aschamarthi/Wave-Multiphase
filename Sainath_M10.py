#!/usr/bin/env python3
"""
Run:  python *.py [--cpu | --cuda]
"""

import warp as wp
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, time as tm, argparse

# ── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--cpu",  action="store_true")
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--restart", type=str, default="")
parser.add_argument("--restart-time", type=float, default=None)
parser.add_argument("--restart-step", type=int, default=None)
args, _ = parser.parse_known_args()
DEVICE = "cpu" if args.cpu else "cuda"

wp.init()

# ── Grid ─────────────────────────────────────────────────────────────────────
NX, NY, G = 1920, 1280, 5
NXG, NYG  = NX + 2*G, NY + 2*G

XMIN_V, XMAX_V = 0.0, 0.030
YMIN_V, YMAX_V = 0.0, 0.020
DX_V = (XMAX_V - XMIN_V) / NX
DY_V = (YMAX_V - YMIN_V) / NY

CFL_NUM  = 0.4
T_END_V  = 16e-6
T_SAVES  = [2e-6, 4e-6, 8e-6, 16e-6]
USE_EMERGENCY_REPAIR = True  # Set False for strictly conservative PP-flux-only runs.

# ── Warp constants ────────────────────────────────────────────────────────────
WG1    = wp.constant(wp.float64(6.12))      # gamma_1  water
WP1    = wp.constant(wp.float64(3.43e8))    # pi_1     water
WG2    = wp.constant(wp.float64(1.4))       # gamma_2  air
WP2    = wp.constant(wp.float64(0.0))       # pi_2     air
WTINY  = wp.constant(wp.float64(1.0e-20))
WEPSM  = wp.constant(wp.float64(1.0e-20))
WMP5C  = wp.constant(wp.float64(4.0))       # MP5 alpha constant
WB2    = wp.constant(wp.float64(4.0 / 3.0))
WKAP   = wp.constant(wp.float64(1.0 / 3.0))
W1o60  = wp.constant(wp.float64(1.0 / 60.0))
W0p5   = wp.constant(wp.float64(0.5))
W1     = wp.constant(wp.float64(1.0))
W2     = wp.constant(wp.float64(2.0))
W3     = wp.constant(wp.float64(3.0))
W13    = wp.constant(wp.float64(13.0))
W27    = wp.constant(wp.float64(27.0))
W47    = wp.constant(wp.float64(47.0))
W4     = wp.constant(wp.float64(4.0))
W0p25  = wp.constant(wp.float64(0.25))
WLIQ   = wp.constant(wp.float64(2.0))       # p_inf threshold for liquid
WBETA_THINC   = wp.constant(wp.float64(1.8))
WYUXIN        = wp.constant(wp.float64(0.9*0.35*1.0e-2 / (1.0 - 0.9*0.35)))  # ≈ 2.903e-3
WTHINC_THRESH = wp.constant(wp.float64(0.35))
WPP_RHO       = wp.constant(wp.float64(1.0e-10))
WPP_ALPHA     = wp.constant(wp.float64(1.0e-10))
WPP_CTILDE    = wp.constant(wp.float64(1.0e-8))
WPP_HS_RHO    = wp.constant(wp.float64(1.0e-11))
WPP_HS_ALPHA  = wp.constant(wp.float64(1.0e-11))
WPP_HS_CTILDE = wp.constant(wp.float64(1.0e-9))

# ── 6-component state struct ──────────────────────────────────────────────────
@wp.struct
class S6:
    v0: wp.float64
    v1: wp.float64
    v2: wp.float64
    v3: wp.float64
    v4: wp.float64
    v5: wp.float64

# ── EOS ───────────────────────────────────────────────────────────────────────
@wp.func
def mix_gamma_pibig(a1: wp.float64):
    a2  = W1 - a1
    gb  = a1 / (WG1 - W1) + a2 / (WG2 - W1)
    pb  = a1 * WG1 * WP1 / (WG1 - W1) + a2 * WG2 * WP2 / (WG2 - W1)
    return gb, pb

@wp.func
def mix_gamma_pinf(a1: wp.float64):
    gb, pb = mix_gamma_pibig(a1)
    g   = W1 / gb + W1
    pi  = (g - W1) * pb / g
    return g, pi

@wp.func
def sound_speed(rho: wp.float64, p: wp.float64, a1: wp.float64) -> wp.float64:
    g, pi = mix_gamma_pinf(a1)
    val = g * (p + pi) / rho
    if val < WTINY:
        val = WTINY
    return wp.sqrt(val)

@wp.func
def prim_to_energy(ar1: wp.float64, ar2: wp.float64, u: wp.float64,
                   v: wp.float64, p: wp.float64, a1: wp.float64) -> wp.float64:
    rho = ar1 + ar2
    gb, pb = mix_gamma_pibig(a1)
    return gb * p + pb + W0p5 * rho * (u*u + v*v)

@wp.func
def clamp01_pp(a1: wp.float64) -> wp.float64:
    a = a1
    if wp.isnan(a):
        a = WPP_ALPHA
    if a < WPP_ALPHA:
        a = WPP_ALPHA
    if a > W1 - WPP_ALPHA:
        a = W1 - WPP_ALPHA
    return a

@wp.func
def pp_theta_lower(base: wp.float64, high: wp.float64, floor: wp.float64) -> wp.float64:
    theta = W1
    if wp.isnan(base) or wp.isnan(high):
        theta = wp.float64(0.0)
    elif high < floor:
        denom = base - high
        if denom > WTINY:
            theta = (base - floor) / denom
        else:
            theta = wp.float64(0.0)
    if theta < wp.float64(0.0):
        theta = wp.float64(0.0)
    if theta > W1:
        theta = W1
    return theta

@wp.func
def pp_theta_upper(base: wp.float64, high: wp.float64, ceil_: wp.float64) -> wp.float64:
    theta = W1
    if wp.isnan(base) or wp.isnan(high):
        theta = wp.float64(0.0)
    elif high > ceil_:
        denom = high - base
        if denom > WTINY:
            theta = (ceil_ - base) / denom
        else:
            theta = wp.float64(0.0)
    if theta < wp.float64(0.0):
        theta = wp.float64(0.0)
    if theta > W1:
        theta = W1
    return theta

@wp.func
def blend_s6(base: S6, high: S6, theta: wp.float64) -> S6:
    s = S6()
    s.v0 = base.v0 + theta * (high.v0 - base.v0)
    s.v1 = base.v1 + theta * (high.v1 - base.v1)
    s.v2 = base.v2 + theta * (high.v2 - base.v2)
    s.v3 = base.v3 + theta * (high.v3 - base.v3)
    s.v4 = base.v4 + theta * (high.v4 - base.v4)
    s.v5 = base.v5 + theta * (high.v5 - base.v5)
    return s

@wp.func
def p_plus_pi(s: S6) -> wp.float64:
    a1 = clamp01_pp(s.v5)
    g, pi = mix_gamma_pinf(a1)
    return s.v4 + pi

@wp.func
def rho_ctilde2_sc(s: S6) -> wp.float64:
    a1 = clamp01_pp(s.v5)
    g, pi = mix_gamma_pinf(a1)
    return (s.v4 + pi) / (g - W1)

@wp.func
def pp_floor_pressure_state(s: S6) -> S6:
    a1 = clamp01_pp(s.v5)
    g, pi = mix_gamma_pinf(a1)
    if wp.isnan(s.v4) or rho_ctilde2_sc(s) < WPP_CTILDE:
        s.v4 = WPP_CTILDE * (g - W1) - pi
    s.v5 = a1
    return s

@wp.func
def wong_stage_lower_s6(base: S6, high: S6, base_val: wp.float64,
                        high_val: wp.float64, floor: wp.float64) -> S6:
    theta = W1
    if wp.isnan(base_val) or wp.isnan(high_val) or base_val < floor:
        theta = wp.float64(0.0)
    elif high_val < floor:
        denom = base_val - high_val
        if denom > WTINY:
            theta = (base_val - floor) / denom
        else:
            theta = wp.float64(0.0)
    return blend_s6(base, high, theta)

@wp.func
def wong_stage_upper_s6(base: S6, high: S6, base_val: wp.float64,
                        high_val: wp.float64, ceil_: wp.float64) -> S6:
    theta = W1
    if wp.isnan(base_val) or wp.isnan(high_val) or base_val > ceil_:
        theta = wp.float64(0.0)
    elif high_val > ceil_:
        denom = high_val - base_val
        if denom > WTINY:
            theta = (ceil_ - base_val) / denom
        else:
            theta = wp.float64(0.0)
    return blend_s6(base, high, theta)

@wp.func
def pp_limit_reconstructed_state(high: S6, base: S6) -> S6:
    # Wong et al. staged interpolation limiter, adapted to the SC reconstructed
    # state [m1, m2, rho*u, rho*v, p, alpha1].
    b = base
    b.v5 = clamp01_pp(b.v5)
    if b.v0 < WPP_RHO or wp.isnan(b.v0):
        b.v0 = WPP_RHO
    if b.v1 < WPP_RHO or wp.isnan(b.v1):
        b.v1 = WPP_RHO

    s = high
    s = wong_stage_lower_s6(b, s, b.v0, s.v0, WPP_RHO)
    s = wong_stage_lower_s6(b, s, b.v1, s.v1, WPP_RHO)
    s = wong_stage_lower_s6(b, s, b.v5, s.v5, WPP_ALPHA)
    s = wong_stage_lower_s6(b, s, W1 - b.v5, W1 - s.v5, WPP_ALPHA)
    s = wong_stage_lower_s6(b, s, rho_ctilde2_sc(b), rho_ctilde2_sc(s), WPP_CTILDE)

    if s.v0 < WPP_HS_RHO or s.v1 < WPP_HS_RHO or \
       s.v5 < WPP_HS_ALPHA or s.v5 > W1 - WPP_HS_ALPHA:
        s = b
    if rho_ctilde2_sc(s) < WPP_HS_CTILDE or wp.isnan(s.v4):
        s = b
    return s

@wp.func
def rho_ctilde2_cons(m1: wp.float64, m2: wp.float64,
                     mx: wp.float64, my: wp.float64,
                     E: wp.float64, a1_in: wp.float64) -> wp.float64:
    a1 = clamp01_pp(a1_in)
    mm1 = m1
    mm2 = m2
    if mm1 < WPP_RHO or wp.isnan(mm1):
        mm1 = WPP_RHO
    if mm2 < WPP_RHO or wp.isnan(mm2):
        mm2 = WPP_RHO
    rho = mm1 + mm2
    uu = mx / rho
    vv = my / rho
    kin = W0p5 * rho * (uu*uu + vv*vv)
    g, pi = mix_gamma_pinf(a1)
    return E - kin - pi

@wp.func
def pressure_margin_cons(m1: wp.float64, m2: wp.float64,
                         mx: wp.float64, my: wp.float64,
                         E: wp.float64, a1_in: wp.float64) -> wp.float64:
    a1 = clamp01_pp(a1_in)
    g, pi = mix_gamma_pinf(a1)
    return rho_ctilde2_cons(m1, m2, mx, my, E, a1) * (g - W1)

@wp.func
def theta_cons_stage(l0: wp.float64, l1: wp.float64,
                     l2: wp.float64, l3: wp.float64,
                     l4: wp.float64, l5: wp.float64,
                     h0: wp.float64, h1: wp.float64,
                     h2: wp.float64, h3: wp.float64,
                     h4: wp.float64, h5: wp.float64,
                     stage: int) -> wp.float64:
    theta = W1
    if stage == 0:
        theta = wp.min(theta, pp_theta_lower(l0, h0, WPP_RHO))
        theta = wp.min(theta, pp_theta_lower(l1, h1, WPP_RHO))
    elif stage == 1:
        theta = wp.min(theta, pp_theta_lower(l5, h5, WPP_ALPHA))
        theta = wp.min(theta, pp_theta_lower(W1 - l5, W1 - h5, WPP_ALPHA))
    else:
        lc = rho_ctilde2_cons(l0, l1, l2, l3, l4, l5)
        hc = rho_ctilde2_cons(h0, h1, h2, h3, h4, h5)
        theta = pp_theta_lower(lc, hc, WPP_CTILDE)

    if theta < wp.float64(0.0) or wp.isnan(theta):
        theta = wp.float64(0.0)
    if theta > W1:
        theta = W1
    return theta

# ── Minmod ────────────────────────────────────────────────────────────────────
@wp.func
def mm2(x: wp.float64, y: wp.float64) -> wp.float64:
    return W0p5 * (wp.sign(x) + wp.sign(y)) * wp.min(wp.abs(x), wp.abs(y))

@wp.func
def mm4(a: wp.float64, b: wp.float64, c: wp.float64, d: wp.float64) -> wp.float64:
    s = wp.sign(a) + wp.sign(b)
    return wp.float64(0.125) * s * wp.abs(
        (wp.sign(a) + wp.sign(c)) * (wp.sign(a) + wp.sign(d))) * \
        wp.min(wp.min(wp.abs(a), wp.abs(b)), wp.min(wp.abs(c), wp.abs(d)))

 # ── Entropy ────────────────────────────────────────────────────────────────────
   
@wp.func
def cell_entropy(ar1: wp.float64, ar2: wp.float64,
                 p: wp.float64, a1: wp.float64) -> wp.float64:
    rho = ar1 + ar2
    if rho < WTINY: rho = WTINY
    g, pi = mix_gamma_pinf(a1)
    return p / wp.pow(rho, g)


# ── Indicator ────────────────────────────────────────────────────────────────────
@wp.func
def cator2_val(em2: wp.float64, em1: wp.float64, e0: wp.float64,
               ep1: wp.float64, ep2: wp.float64) -> wp.float64:
    aa = wp.float64(0.25)*wp.abs(em2 - wp.float64(4.0)*em1 + wp.float64(3.0)*e0) \
       + (wp.float64(13.0)/wp.float64(12.0))*wp.abs(e0 - wp.float64(2.0)*em1 + em2)
    bb = wp.float64(0.25)*wp.abs(wp.float64(3.0)*e0 - wp.float64(4.0)*ep1 + ep2) \
       + (wp.float64(13.0)/wp.float64(12.0))*wp.abs(e0 - wp.float64(2.0)*ep1 + ep2)
    return (wp.float64(2.0)*aa*bb + WYUXIN) / (aa*aa + bb*bb + WYUXIN)

@wp.func
def thinc_ul(vm1: wp.float64, v0: wp.float64, v1: wp.float64) -> wp.float64:
    result = v0
    if (v1 - v0)*(v0 - vm1) > wp.float64(0.0):
        qa   = W0p5*(v1 + vm1)
        qd   = W0p5*(v1 - vm1)
        feng = (v0 - qa) / (qd)
        T1   = wp.tanh(WBETA_THINC * W0p5)
        T2   = wp.tanh(feng * WBETA_THINC * W0p5)
        result = qa + qd*(T1 + T2/T1)/(W1 + T2)
    return result

@wp.func
def thinc_ur(v0: wp.float64, v1: wp.float64, v2: wp.float64) -> wp.float64:
    result = v1
    if (v2 - v1)*(v1 - v0) > wp.float64(0.0):
        qa   = W0p5*(v2 + v0)
        qd   = W0p5*(v2 - v0)
        feng = (v1 - qa) / (qd)
        T1   = wp.tanh(WBETA_THINC * W0p5)
        T2   = wp.tanh(feng * WBETA_THINC * W0p5)
        result = qa - qd*(T1 - T2/T1)/(W1 - T2)
    return result

# ── MP5 reconstruction (single characteristic variable) ───────────────────────
# Stencil convention: vm2=ix-2, vm1=ix-1, v0=ix, vp1=ix+1, vp2=ix+2, vp3=ix+3
# ul = left-biased state at interface (from cell v0 side)
# ur = right-biased state at interface (from cell vp1 side)

@wp.func
def mp5_ul(vm2: wp.float64, vm1: wp.float64, v0: wp.float64,
           vp1: wp.float64, vp2: wp.float64) -> wp.float64:
    VOR = W1o60 * (W2*vm2 - W13*vm1 + W47*v0 + W27*vp1 - W3*vp2)
    VMP = v0 + mm2(vp1 - v0, WMP5C * (v0 - vm1))
    result = VOR
    if (VOR - v0) * (VOR - VMP) >= WEPSM:
        DJM1   = vm2 - W2*vm1 + v0
        DJ     = vm1 - W2*v0  + vp1
        DJP1   = v0  - W2*vp1 + vp2
        DM4JPH = mm4(W4*DJ - DJP1, W4*DJP1 - DJ, DJ, DJP1)
        DM4JMH = mm4(W4*DJ - DJM1, W4*DJM1 - DJ, DJ, DJM1)
        VUL    = v0  + WMP5C * (v0 - vm1)
        VAV    = W0p5 * (v0 + vp1)
        VMD    = VAV - W0p5 * DM4JPH
        VLC    = v0  + W0p5 * (v0 - vm1) + WB2 * DM4JMH
        VMIN   = wp.max(wp.min(wp.min(v0, vp1), VMD), wp.min(wp.min(v0, VUL), VLC))
        VMAX   = wp.min(wp.max(wp.max(v0, vp1), VMD), wp.max(wp.max(v0, VUL), VLC))
        a_     = v0  - vm1
        b_     = vp1 - v0
        u_re   = v0 + (wp.sign(a_) + wp.sign(b_)) * W0p5 * \
                 (wp.abs(a_) * wp.abs(b_)) / (wp.abs(a_) + wp.abs(b_) + wp.float64(1.0e-20))
        result = W0p5*(VOR + u_re) - wp.sign((VOR - VMIN)*(VOR - VMAX)) * W0p5*(VOR - u_re)
        result = VOR + mm2(VMIN - VOR, VMAX - VOR)   # <-- correct

    return result

@wp.func
def mp5_ur(vm1: wp.float64, v0: wp.float64, vp1: wp.float64,
           vp2: wp.float64, vp3: wp.float64) -> wp.float64:

    VOR = W1o60 * (wp.float64(-3.0)*vm1 + W27*v0 + W47*vp1 - W13*vp2 + W2*vp3)
    VMP = vp1 + mm2(v0 - vp1, WMP5C * (vp1 - vp2))
    result = VOR
    if (VOR - vp1) * (VOR - VMP) >= WEPSM:
        DJM1   = vm1 - W2*v0  + vp1
        DJ     = v0  - W2*vp1 + vp2
        DJP1   = vp1 - W2*vp2 + vp3
        DM4JPH = mm4(W4*DJ - DJP1, W4*DJP1 - DJ, DJ, DJP1)
        DM4JMH = mm4(W4*DJ - DJM1, W4*DJM1 - DJ, DJ, DJM1)
        VUL    = vp1 + WMP5C * (vp1 - vp2)
        VAV    = W0p5 * (vp1 + v0)
        VMD    = VAV - W0p5 * DM4JMH
        VLC    = vp1 + W0p5 * (vp1 - vp2) + WB2 * DM4JPH
        VMIN   = wp.max(wp.min(wp.min(vp1, v0), VMD), wp.min(wp.min(vp1, VUL), VLC))
        VMAX   = wp.min(wp.max(wp.max(vp1, v0), VMD), wp.max(wp.max(vp1, VUL), VLC))
        a_     = vp1 - v0
        b_     = vp2 - vp1
        u_re   = vp1 + (wp.sign(v0 - vp1) + wp.sign(vp1 - vp2)) * W0p5 * \
                 (wp.abs(a_) * wp.abs(b_)) / (wp.abs(a_) + wp.abs(b_) + wp.float64(1.0e-20))
        result = W0p5*(VOR + u_re) - wp.sign((VOR - VMIN)*(VOR - VMAX)) * W0p5*(VOR - u_re)
        result = VOR + mm2(VMIN - VOR, VMAX - VOR)
    return result

@wp.func
def central6_avg(vm2: wp.float64, vm1: wp.float64, v0: wp.float64,
                 vp1: wp.float64, vp2: wp.float64, vp3: wp.float64) -> wp.float64:
    left  = W1o60 * (W2*vm2 - W13*vm1 + W47*v0 + W27*vp1 - W3*vp2)
    right = W1o60 * (wp.float64(-3.0)*vm1 + W27*v0 + W47*vp1 - W13*vp2 + W2*vp3)
    return W0p5 * (left + right)

@wp.func
def muscl_ul(vm1: wp.float64, v0: wp.float64, vp1: wp.float64, vp2: wp.float64) -> wp.float64:
    dm = v0  - vm1
    d0 = vp1 - v0
    dp = vp2 - vp1
    return v0 + W0p25 * ((W1 - WKAP)*mm2(dm, W2*d0) + (W1 + WKAP)*mm2(d0, W2*dm))

@wp.func
def muscl_ur(vm1: wp.float64, v0: wp.float64, vp1: wp.float64, vp2: wp.float64) -> wp.float64:
    dm = v0  - vm1
    d0 = vp1 - v0
    dp = vp2 - vp1
    return vp1 - W0p25 * ((W1 - WKAP)*mm2(dp, W2*d0) + (W1 + WKAP)*mm2(d0, W2*dp))


@wp.func
def muscl_char_ul(vm1: wp.float64, v0: wp.float64,
                  vp1: wp.float64, vp2: wp.float64) -> wp.float64:
    dm = v0  - vm1
    d0 = vp1 - v0
    return v0 + wp.float64(0.25) * (
        (W1 - WKAP) * mm2(dm, W2 * d0) +
        (W1 + WKAP) * mm2(d0, W2 * dm))

@wp.func
def muscl_char_ur(vm1: wp.float64, v0: wp.float64,
                  vp1: wp.float64, vp2: wp.float64) -> wp.float64:
    dp = vp2 - vp1
    d0 = vp1 - v0
    return vp1 - wp.float64(0.25) * (
        (W1 - WKAP) * mm2(dp, W2 * d0) +
        (W1 + WKAP) * mm2(d0, W2 * dp))


@wp.func
def to_char_x(u0: wp.float64, u1: wp.float64, u2: wp.float64,
              u3: wp.float64, u4: wp.float64, u5: wp.float64,
              c: wp.float64, rho: wp.float64,
              ar1: wp.float64, ar2: wp.float64,
              u_avg: wp.float64, v_avg: wp.float64) -> S6:  # ADD THESE
    inv2c  = W0p5 / c
    inv2c2 = W0p5 / (c * c)
    Y1 = ar1 / rho
    Y2 = ar2 / rho
    un = u_avg   
    ut = v_avg   
    s = S6()
    s.v0 =  un*inv2c*u0 + un*inv2c*u1 - inv2c*u2 + inv2c2*u4
    s.v1 =  u0 - Y1/(c*c)*u4
    s.v2 =  u1 - Y2/(c*c)*u4
    s.v3 = -ut*u0 - ut*u1 + u3
    s.v4 =  u5
    s.v5 = -un*inv2c*u0 - un*inv2c*u1 + inv2c*u2 + inv2c2*u4
    return s




@wp.func
def to_phys_x(w: S6, c: wp.float64, rho: wp.float64,
              ar1: wp.float64, ar2: wp.float64,
              u_avg: wp.float64, v_avg: wp.float64) -> S6:

    c2   = c * c
    Y1   = ar1 / rho
    Y2   = ar2 / rho

    s = S6()
    # ar1:  R col0*w0 + R col1*w1 + R col5*w5  (Y1 appears in acoustic cols)
    s.v0 = Y1 * (w.v0 + w.v5) + w.v1
    # ar2:
    s.v1 = Y2 * (w.v0 + w.v5) + w.v2
    # rho*u (x-momentum):
    #   col0: (u-c)*w0, col1: u*w1, col2: u*w2, col3: 0 (x-dir -my=0),
    #   col5: (u+c)*w5
    s.v2 = (u_avg - c) * w.v0 + u_avg * w.v1 + u_avg * w.v2 + (u_avg + c) * w.v5
    # rho*v (y-momentum):
    #   col0: v*w0, col1: v*w1, col2: v*w2, col3: mx*w3=w3, col5: v*w5
    s.v3 = v_avg * (w.v0 + w.v1 + w.v2 + w.v5) + w.v3
    # p:
    #   col0: c2*w0, col5: c2*w5  (all other cols have 0 in pressure row)
    s.v4 = c2 * (w.v0 + w.v5)
    # alpha1:
    s.v5 = w.v4
    return s


@wp.func
def to_char_y(u0: wp.float64, u1: wp.float64, u2: wp.float64,
              u3: wp.float64, u4: wp.float64, u5: wp.float64,
              c: wp.float64, rho: wp.float64,
              ar1: wp.float64, ar2: wp.float64,
              u_avg: wp.float64, v_avg: wp.float64) -> S6:  # ADD THESE
    inv2c  = W0p5 / c
    inv2c2 = W0p5 / (c * c)
    Y1 = ar1 / rho
    Y2 = ar2 / rho
    un   =  v_avg        
    ut_f = -u_avg        
    s = S6()
    s.v0 =  un*inv2c*u0 + un*inv2c*u1 - inv2c*u3 + inv2c2*u4
    s.v1 =  u0 - Y1/(c*c)*u4
    s.v2 =  u1 - Y2/(c*c)*u4
    s.v3 = -ut_f*u0 - ut_f*u1 - u2
    s.v4 =  u5
    s.v5 = -un*inv2c*u0 - un*inv2c*u1 + inv2c*u3 + inv2c2*u4
    return s


@wp.func
def to_phys_y(w: S6, c: wp.float64, rho: wp.float64,
              ar1: wp.float64, ar2: wp.float64,
              u_avg: wp.float64, v_avg: wp.float64) -> S6:

    c2   = c * c
    Y1   = ar1 / rho
    Y2   = ar2 / rho

    s = S6()
    # ar1
    s.v0 = Y1 * (w.v0 + w.v5) + w.v1
    # ar2
    s.v1 = Y2 * (w.v0 + w.v5) + w.v2
    # rho*u (x-momentum, tangential y-dir):
    #   col0: u*w0, col1: u*w1, col2: u*w2,
    #   col3: -my*w3 = -1*w3,  col5: u*w5
    s.v2 = u_avg * (w.v0 + w.v1 + w.v2 + w.v5) - w.v3
    # rho*v (y-momentum, normal y-dir):
    #   col0: (v-c)*w0, col1: v*w1, col2: v*w2, col3: mx*w3=0, col5: (v+c)*w5
    s.v3 = (v_avg - c) * w.v0 + v_avg * w.v1 + v_avg * w.v2 + (v_avg + c) * w.v5
    # p
    s.v4 = c2 * (w.v0 + w.v5)
    # alpha1
    s.v5 = w.v4
    return s


# ── HLLC flux (x-direction) ────────────────────────────────────────────────────
@wp.func
def hllc_x(ar1L: wp.float64, ar2L: wp.float64, uL: wp.float64, vL: wp.float64,
           pL:   wp.float64, a1L:  wp.float64,
           ar1R: wp.float64, ar2R: wp.float64, uR: wp.float64, vR: wp.float64,
           pR:   wp.float64, a1R:  wp.float64) -> S6:
    rhoL = ar1L + ar2L
    rhoR = ar1R + ar2R
    cL   = sound_speed(rhoL, pL, a1L)
    cR   = sound_speed(rhoR, pR, a1R)
    EL   = prim_to_energy(ar1L, ar2L, uL, vL, pL, a1L)
    ER   = prim_to_energy(ar1R, ar2R, uR, vR, pR, a1R)

    u_avg = W0p5 * (uL + uR)
    c_avg = W0p5 * (cL + cR)
    SL    = wp.min(uL - cL, u_avg - c_avg)
    SR    = wp.max(uR + cR, u_avg + c_avg)

    denom = rhoL*(SL - uL) - rhoR*(SR - uR)
    if wp.abs(denom) < WTINY:
        denom = WTINY
    SP = (pR - pL + rhoL*uL*(SL - uL) - rhoR*uR*(SR - uR)) / denom

    f = S6()

    if SL > wp.float64(0.0):
        f.v0 = ar1L * uL
        f.v1 = ar2L * uL
        f.v2 = rhoL * uL*uL + pL
        f.v3 = rhoL * uL * vL
        f.v4 = (EL + pL) * uL
        f.v5 = a1L * uL
    elif SL <= wp.float64(0.0) and wp.float64(0.0) < SP:
        ratio = (SL - uL) / (SL - SP)
        EL_star = EL + (SP - uL) * (SP*rhoL + pL/(SL - uL))
        fL0 = ar1L * uL;  fL1 = ar2L * uL
        fL2 = rhoL*uL*uL + pL;  fL3 = rhoL*uL*vL
        fL4 = (EL + pL)*uL;     fL5 = a1L*uL
        f.v0 = fL0 + SL * (ar1L  * ratio - ar1L)
        f.v1 = fL1 + SL * (ar2L  * ratio - ar2L)
        f.v2 = fL2 + SL * (rhoL*SP * ratio - rhoL*uL)
        f.v3 = fL3 + SL * (rhoL*vL * ratio - rhoL*vL)
        f.v4 = fL4 + SL * (EL_star * ratio - EL)
        f.v5 = fL5 + SL * (a1L  * ratio - a1L)
        # Interface velocity for non-conservative source (left side)
        ustar = uL + wp.min(wp.float64(0.0), SL) * (ratio - W1)
        f.v5 = fL5 + SL * (a1L * ratio - a1L)  # flux alpha_1
        # Pack ustar into a separate field — we use v5 for alpha flux and
        # return it; source handled in kernel
    elif SP <= wp.float64(0.0) and wp.float64(0.0) <= SR:
        ratio = (SR - uR) / (SR - SP)
        ER_star = ER + (SP - uR) * (SP*rhoR + pR/(SR - uR))
        fR0 = ar1R*uR;  fR1 = ar2R*uR
        fR2 = rhoR*uR*uR + pR;  fR3 = rhoR*uR*vR
        fR4 = (ER + pR)*uR;     fR5 = a1R*uR
        f.v0 = fR0 + SR * (ar1R  * ratio - ar1R)
        f.v1 = fR1 + SR * (ar2R  * ratio - ar2R)
        f.v2 = fR2 + SR * (rhoR*SP * ratio - rhoR*uR)
        f.v3 = fR3 + SR * (rhoR*vR * ratio - rhoR*vR)
        f.v4 = fR4 + SR * (ER_star * ratio - ER)
        f.v5 = fR5 + SR * (a1R  * ratio - a1R)
    else:
        f.v0 = ar1R * uR
        f.v1 = ar2R * uR
        f.v2 = rhoR * uR*uR + pR
        f.v3 = rhoR * uR * vR
        f.v4 = (ER + pR) * uR
        f.v5 = a1R * uR

    return f

@wp.func
def hllc_x_src(ar1L: wp.float64, ar2L: wp.float64, uL: wp.float64,
               pL:   wp.float64, a1L:  wp.float64,
               ar1R: wp.float64, ar2R: wp.float64, uR: wp.float64,
               pR:   wp.float64, a1R:  wp.float64) -> wp.float64:
    """Interface velocity u★ for non-conservative α₁ source term (x-dir)."""
    rhoL = ar1L + ar2L
    rhoR = ar1R + ar2R
    cL   = sound_speed(rhoL, pL, a1L)
    cR   = sound_speed(rhoR, pR, a1R)
    u_avg = W0p5*(uL + uR);  c_avg = W0p5*(cL + cR)
    SL = wp.min(uL - cL, u_avg - c_avg)
    SR = wp.max(uR + cR, u_avg + c_avg)
    denom = rhoL*(SL - uL) - rhoR*(SR - uR)
    if wp.abs(denom) < WTINY:
        denom = WTINY
    SP = (pR - pL + rhoL*uL*(SL - uL) - rhoR*uR*(SR - uR)) / denom

    sL_m_uL = SL - uL;  sR_m_uR = SR - uR
    sL_m_SP = SL - SP;  sR_m_SP = SR - SP
    left_part  = uL + wp.min(wp.float64(0.0), SL) * (sL_m_uL/sL_m_SP - W1)
    right_part = uR + wp.max(wp.float64(0.0), SR) * (sR_m_uR/sR_m_SP - W1)
    sp_sign = wp.sign(SP)
    return W0p5*(W1 + sp_sign)*left_part + W0p5*(W1 - sp_sign)*right_part

@wp.func
def hllc_y(ar1L: wp.float64, ar2L: wp.float64, uL: wp.float64, vL: wp.float64,
           pL:   wp.float64, a1L:  wp.float64,
           ar1R: wp.float64, ar2R: wp.float64, uR: wp.float64, vR: wp.float64,
           pR:   wp.float64, a1R:  wp.float64) -> S6:
    rhoL = ar1L + ar2L
    rhoR = ar1R + ar2R
    cL   = sound_speed(rhoL, pL, a1L)
    cR   = sound_speed(rhoR, pR, a1R)
    EL   = prim_to_energy(ar1L, ar2L, uL, vL, pL, a1L)
    ER   = prim_to_energy(ar1R, ar2R, uR, vR, pR, a1R)

    v_avg = W0p5*(vL + vR);  c_avg = W0p5*(cL + cR)
    SL = wp.min(vL - cL, v_avg - c_avg)
    SR = wp.max(vR + cR, v_avg + c_avg)

    denom = rhoL*(SL - vL) - rhoR*(SR - vR)
    if wp.abs(denom) < WTINY:
        denom = WTINY
    SP = (pR - pL + rhoL*vL*(SL - vL) - rhoR*vR*(SR - vR)) / denom

    f = S6()

    if SL > wp.float64(0.0):
        f.v0 = ar1L * vL
        f.v1 = ar2L * vL
        f.v2 = rhoL * uL * vL
        f.v3 = rhoL * vL*vL + pL
        f.v4 = (EL + pL) * vL
        f.v5 = a1L * vL
    elif SL <= wp.float64(0.0) and wp.float64(0.0) < SP:
        ratio = (SL - vL) / (SL - SP)
        EL_star = EL + (SP - vL) * (SP*rhoL + pL/(SL - vL))
        fL0 = ar1L*vL;  fL1 = ar2L*vL
        fL2 = rhoL*uL*vL;  fL3 = rhoL*vL*vL + pL
        fL4 = (EL + pL)*vL;  fL5 = a1L*vL
        f.v0 = fL0 + SL * (ar1L   * ratio - ar1L)
        f.v1 = fL1 + SL * (ar2L   * ratio - ar2L)
        f.v2 = fL2 + SL * (rhoL*uL * ratio - rhoL*uL)
        f.v3 = fL3 + SL * (rhoL*SP * ratio - rhoL*vL)
        f.v4 = fL4 + SL * (EL_star * ratio - EL)
        f.v5 = fL5 + SL * (a1L    * ratio - a1L)
    elif SP <= wp.float64(0.0) and wp.float64(0.0) <= SR:
        ratio = (SR - vR) / (SR - SP)
        ER_star = ER + (SP - vR) * (SP*rhoR + pR/(SR - vR))
        fR0 = ar1R*vR;  fR1 = ar2R*vR
        fR2 = rhoR*uR*vR;  fR3 = rhoR*vR*vR + pR
        fR4 = (ER + pR)*vR;  fR5 = a1R*vR
        f.v0 = fR0 + SR * (ar1R   * ratio - ar1R)
        f.v1 = fR1 + SR * (ar2R   * ratio - ar2R)
        f.v2 = fR2 + SR * (rhoR*uR * ratio - rhoR*uR)
        f.v3 = fR3 + SR * (rhoR*SP * ratio - rhoR*vR)
        f.v4 = fR4 + SR * (ER_star * ratio - ER)
        f.v5 = fR5 + SR * (a1R    * ratio - a1R)
    else:
        f.v0 = ar1R * vR
        f.v1 = ar2R * vR
        f.v2 = rhoR * uR * vR
        f.v3 = rhoR * vR*vR + pR
        f.v4 = (ER + pR) * vR
        f.v5 = a1R * vR

    return f

@wp.func
def hllc_y_src(ar1L: wp.float64, ar2L: wp.float64, vL: wp.float64,
               pL:   wp.float64, a1L:  wp.float64,
               ar1R: wp.float64, ar2R: wp.float64, vR: wp.float64,
               pR:   wp.float64, a1R:  wp.float64) -> wp.float64:
    rhoL = ar1L + ar2L;  rhoR = ar1R + ar2R
    cL   = sound_speed(rhoL, pL, a1L);  cR = sound_speed(rhoR, pR, a1R)
    v_avg = W0p5*(vL + vR);  c_avg = W0p5*(cL + cR)
    SL = wp.min(vL - cL, v_avg - c_avg)
    SR = wp.max(vR + cR, v_avg + c_avg)
    denom = rhoL*(SL - vL) - rhoR*(SR - vR)
    if wp.abs(denom) < WTINY:
        denom = WTINY
    SP = (pR - pL + rhoL*vL*(SL - vL) - rhoR*vR*(SR - vR)) / denom
    left_part  = vL + wp.min(wp.float64(0.0), SL) * ((SL-vL)/(SL-SP) - W1)
    right_part = vR + wp.max(wp.float64(0.0), SR) * ((SR-vR)/(SR-SP) - W1)
    sp_sign = wp.sign(SP)
    return W0p5*(W1 + sp_sign)*left_part + W0p5*(W1 - sp_sign)*right_part

# ══════════════════════════════════════════════════════════════════════════════
# Kernels
# ══════════════════════════════════════════════════════════════════════════════

# Separate 2D arrays per variable: ar1, ar2, ru, rv, E, al  plus u_p, v_p, p_p

@wp.kernel
def ic_kernel2(ar1: wp.array2d(dtype=wp.float64),
               ar2: wp.array2d(dtype=wp.float64),
               ru:  wp.array2d(dtype=wp.float64),
               rv:  wp.array2d(dtype=wp.float64),
               E:   wp.array2d(dtype=wp.float64),
               al:  wp.array2d(dtype=wp.float64),
               u_p: wp.array2d(dtype=wp.float64),
               v_p: wp.array2d(dtype=wp.float64),
               p_p: wp.array2d(dtype=wp.float64),
               xarr: wp.array(dtype=wp.float64),
               yarr: wp.array(dtype=wp.float64)):
    i, j = wp.tid()
    xi = xarr[i];  yj = yarr[j]

    a1r1 = wp.float64(0.0); a2r2 = wp.float64(0.0)
    uu = wp.float64(0.0);   vv = wp.float64(0.0)
    pp = wp.float64(0.0);   aa = wp.float64(0.0)

    shock_x = wp.float64(0.009)
    cyl_xc  = wp.float64(0.013)
    cyl_yc  = wp.float64(0.010)
    cyl_r   = wp.float64(0.004)

    dx2 = (xi - cyl_xc) * (xi - cyl_xc) \
        + (yj - cyl_yc) * (yj - cyl_yc)

    if dx2 <= cyl_r * cyl_r:
        # Water cylinder: alpha1 is water volume fraction.
        a1r1 = wp.float64(1000.0);  a2r2 = wp.float64(1.0e-8)
        uu   = wp.float64(0.0);     vv   = wp.float64(0.0)
        pp   = wp.float64(1.0e5);   aa   = W1 - wp.float64(1.0e-8)
    elif xi < shock_x:
        # Post-shock air, Mach 10 table.
        a1r1 = wp.float64(1.0e-8);  a2r2 = wp.float64(6.8571)
        uu   = wp.float64(2817.9);  vv   = wp.float64(0.0)
        pp   = wp.float64(1.165e7); aa   = wp.float64(1.0e-8)
    else:
        # Pre-shock air.
        a1r1 = wp.float64(1.0e-8);  a2r2 = wp.float64(1.2)
        uu   = wp.float64(0.0);     vv   = wp.float64(0.0)
        pp   = wp.float64(1.0e5);   aa   = wp.float64(1.0e-8)

    rho  = a1r1 + a2r2
    gb, pb = mix_gamma_pibig(aa)

    ar1[i, j] = a1r1
    ar2[i, j] = a2r2
    ru[i, j]  = rho * uu
    rv[i, j]  = rho * vv
    E[i, j]   = gb*pp + pb + W0p5*rho*(uu*uu + vv*vv)
    al[i, j]  = aa
    u_p[i, j] = uu
    v_p[i, j] = vv
    p_p[i, j] = pp


@wp.kernel
def cons_to_prim_kernel(ar1: wp.array2d(dtype=wp.float64),
                        ar2: wp.array2d(dtype=wp.float64),
                        ru:  wp.array2d(dtype=wp.float64),
                        rv:  wp.array2d(dtype=wp.float64),
                        E:   wp.array2d(dtype=wp.float64),
                        al:  wp.array2d(dtype=wp.float64),
                        u_p: wp.array2d(dtype=wp.float64),
                        v_p: wp.array2d(dtype=wp.float64),
                        p_p: wp.array2d(dtype=wp.float64)):
    i, j = wp.tid()
    a1  = al[i, j]
    if a1 < wp.float64(1.0e-8): a1 = wp.float64(1.0e-8)
    if a1 > W1 - wp.float64(1.0e-8): a1 = W1 - wp.float64(1.0e-8)
    al[i, j] = a1

    rho = ar1[i, j] + ar2[i, j]
    uu  = ru[i, j] / rho
    vv  = rv[i, j] / rho
    gb, pb = mix_gamma_pibig(a1)
    pp  = (E[i, j] - pb - W0p5*rho*(uu*uu + vv*vv)) / gb

    u_p[i, j] = uu
    v_p[i, j] = vv
    p_p[i, j] = pp


@wp.kernel
def enforce_admissible_kernel(ar1: wp.array2d(dtype=wp.float64),
                              ar2: wp.array2d(dtype=wp.float64),
                              ru:  wp.array2d(dtype=wp.float64),
                              rv:  wp.array2d(dtype=wp.float64),
                              E:   wp.array2d(dtype=wp.float64),
                              al:  wp.array2d(dtype=wp.float64)):
    i, j = wp.tid()

    a1 = clamp01_pp(al[i, j])
    m1 = ar1[i, j]
    m2 = ar2[i, j]
    mx = ru[i, j]
    my = rv[i, j]

    if wp.isnan(m1) or m1 < WPP_RHO:
        m1 = WPP_RHO
    if wp.isnan(m2) or m2 < WPP_RHO:
        m2 = WPP_RHO
    if wp.isnan(mx):
        mx = wp.float64(0.0)
    if wp.isnan(my):
        my = wp.float64(0.0)

    rho = m1 + m2
    uu = mx / rho
    vv = my / rho
    kin = W0p5 * rho * (uu*uu + vv*vv)
    gb, pb = mix_gamma_pibig(a1)
    p = (E[i, j] - pb - kin) / gb
    g, pi = mix_gamma_pinf(a1)
    p_min = WPP_CTILDE * (g - W1) - pi

    if wp.isnan(E[i, j]) or wp.isnan(p) or rho_ctilde2_cons(m1, m2, mx, my, E[i, j], a1) < WPP_CTILDE:
        E[i, j] = gb * p_min + pb + kin

    ar1[i, j] = m1
    ar2[i, j] = m2
    ru[i, j] = mx
    rv[i, j] = my
    al[i, j] = a1


# ══════════════════════════════════════════════════════════════════════════════
#  fx_kernel  (replace entire function body)
# ══════════════════════════════════════════════════════════════════════════════


@wp.kernel
def fx_kernel(ar1: wp.array2d(dtype=wp.float64),
              ar2: wp.array2d(dtype=wp.float64),
              ru:  wp.array2d(dtype=wp.float64),   
              rv:  wp.array2d(dtype=wp.float64),   
              p_p: wp.array2d(dtype=wp.float64),
              al:  wp.array2d(dtype=wp.float64),
              fx0: wp.array2d(dtype=wp.float64),
              fx1: wp.array2d(dtype=wp.float64),
              fx2: wp.array2d(dtype=wp.float64),
              fx3: wp.array2d(dtype=wp.float64),
              fx4: wp.array2d(dtype=wp.float64),
              fx5: wp.array2d(dtype=wp.float64),
              srcx: wp.array2d(dtype=wp.float64),
              ofs_i: int, ofs_j: int):

    ii_loc, jj_loc = wp.tid()
    ii = ii_loc + ofs_i
    jj = jj_loc + ofs_j

    pm2 = S6()
    pm2.v0=ar1[ii-2,jj]; pm2.v1=ar2[ii-2,jj]
    pm2.v2=ru[ii-2,jj];  pm2.v3=rv[ii-2,jj]
    pm2.v4=p_p[ii-2,jj]; pm2.v5=al[ii-2,jj]

    pm1 = S6()
    pm1.v0=ar1[ii-1,jj]; pm1.v1=ar2[ii-1,jj]
    pm1.v2=ru[ii-1,jj];  pm1.v3=rv[ii-1,jj]
    pm1.v4=p_p[ii-1,jj]; pm1.v5=al[ii-1,jj]

    p0 = S6()
    p0.v0=ar1[ii,jj]; p0.v1=ar2[ii,jj]
    p0.v2=ru[ii,jj];  p0.v3=rv[ii,jj]
    p0.v4=p_p[ii,jj]; p0.v5=al[ii,jj]

    pp1 = S6()
    pp1.v0=ar1[ii+1,jj]; pp1.v1=ar2[ii+1,jj]
    pp1.v2=ru[ii+1,jj];  pp1.v3=rv[ii+1,jj]
    pp1.v4=p_p[ii+1,jj]; pp1.v5=al[ii+1,jj]

    pp2 = S6()
    pp2.v0=ar1[ii+2,jj]; pp2.v1=ar2[ii+2,jj]
    pp2.v2=ru[ii+2,jj];  pp2.v3=rv[ii+2,jj]
    pp2.v4=p_p[ii+2,jj]; pp2.v5=al[ii+2,jj]

    pp3 = S6()
    pp3.v0=ar1[ii+3,jj]; pp3.v1=ar2[ii+3,jj]
    pp3.v2=ru[ii+3,jj];  pp3.v3=rv[ii+3,jj]
    pp3.v4=p_p[ii+3,jj]; pp3.v5=al[ii+3,jj]

    # ── Entropy and THINC indicator ───────────────────────────────────────────
    e_m3 = cell_entropy(ar1[ii-3,jj], ar2[ii-3,jj], p_p[ii-3,jj], al[ii-3,jj])
    e_m2 = cell_entropy(ar1[ii-2,jj], ar2[ii-2,jj], p_p[ii-2,jj], al[ii-2,jj])
    e_m1 = cell_entropy(ar1[ii-1,jj], ar2[ii-1,jj], p_p[ii-1,jj], al[ii-1,jj])
    e_0  = cell_entropy(ar1[ii+0,jj], ar2[ii+0,jj], p_p[ii+0,jj], al[ii+0,jj])
    e_p1 = cell_entropy(ar1[ii+1,jj], ar2[ii+1,jj], p_p[ii+1,jj], al[ii+1,jj])
    e_p2 = cell_entropy(ar1[ii+2,jj], ar2[ii+2,jj], p_p[ii+2,jj], al[ii+2,jj])
    e_p3 = cell_entropy(ar1[ii+3,jj], ar2[ii+3,jj], p_p[ii+3,jj], al[ii+3,jj])
    e_p4 = cell_entropy(ar1[ii+4,jj], ar2[ii+4,jj], p_p[ii+4,jj], al[ii+4,jj])

    c_m1 = cator2_val(e_m3, e_m2, e_m1, e_0,  e_p1)  # ix-1
    c_0  = cator2_val(e_m2, e_m1, e_0,  e_p1, e_p2)  # ix
    c_p1 = cator2_val(e_m1, e_0,  e_p1, e_p2, e_p3)  # ix+1
    c_p2 = cator2_val(e_0,  e_p1, e_p2, e_p3, e_p4)  # ix+2
    use_thinc = wp.min(wp.min(c_m1, c_0), wp.min(c_p1, c_p2)) < WTHINC_THRESH

    # ── Interface-averaged Roe state ───────────────────────────────────────
    ar1a = W0p5 * (p0.v0 + pp1.v0)
    ar2a = W0p5 * (p0.v1 + pp1.v1)
    a1a  = W0p5 * (p0.v5 + pp1.v5)
    pa   = W0p5 * (p0.v4 + pp1.v4)
    rhoa = ar1a + ar2a
    # interface velocities from momentum
    u_avg = W0p5 * (p0.v2 + pp1.v2) / rhoa   
    v_avg = W0p5 * (p0.v3 + pp1.v3) / rhoa   
    ca    = sound_speed(rhoa, pa, a1a)
    gdum, pinf_a = mix_gamma_pinf(a1a)

    # ── Always characteristic reconstruction ───────────────────────────────
    wm2 = to_char_x(pm2.v0,pm2.v1,pm2.v2,pm2.v3,pm2.v4,pm2.v5, ca,rhoa,ar1a,ar2a, u_avg,v_avg)
    wm1 = to_char_x(pm1.v0,pm1.v1,pm1.v2,pm1.v3,pm1.v4,pm1.v5, ca,rhoa,ar1a,ar2a, u_avg,v_avg)
    w0  = to_char_x(p0.v0, p0.v1, p0.v2, p0.v3, p0.v4, p0.v5,  ca,rhoa,ar1a,ar2a, u_avg,v_avg)
    w1  = to_char_x(pp1.v0,pp1.v1,pp1.v2,pp1.v3,pp1.v4,pp1.v5, ca,rhoa,ar1a,ar2a, u_avg,v_avg)
    w2  = to_char_x(pp2.v0,pp2.v1,pp2.v2,pp2.v3,pp2.v4,pp2.v5, ca,rhoa,ar1a,ar2a, u_avg,v_avg)
    w3  = to_char_x(pp3.v0,pp3.v1,pp3.v2,pp3.v3,pp3.v4,pp3.v5, ca,rhoa,ar1a,ar2a, u_avg,v_avg)

    wL = S6(); wR = S6()


    # MUSCL
    wL.v0 = muscl_char_ul(wm1.v0, w0.v0, w1.v0, w2.v0)
    wL.v1 = muscl_char_ul(wm1.v1, w0.v1, w1.v1, w2.v1)
    wL.v2 = muscl_char_ul(wm1.v2, w0.v2, w1.v2, w2.v2)
    wL.v3 = muscl_char_ul(wm1.v3, w0.v3, w1.v3, w2.v3)
    wL.v4 = muscl_char_ul(wm1.v4, w0.v4, w1.v4, w2.v4)
    wL.v5 = muscl_char_ul(wm1.v5, w0.v5, w1.v5, w2.v5)
    wR.v0 = muscl_char_ur(wm1.v0, w0.v0, w1.v0, w2.v0)
    wR.v1 = muscl_char_ur(wm1.v1, w0.v1, w1.v1, w2.v1)
    wR.v2 = muscl_char_ur(wm1.v2, w0.v2, w1.v2, w2.v2)
    wR.v3 = muscl_char_ur(wm1.v3, w0.v3, w1.v3, w2.v3)
    wR.v4 = muscl_char_ur(wm1.v4, w0.v4, w1.v4, w2.v4)
    wR.v5 = muscl_char_ur(wm1.v5, w0.v5, w1.v5, w2.v5)
    
    # # THINC override for v1, v2, v4 (Fortran i=2,3,5)
    if use_thinc:
        wL.v1 = thinc_ul(wm1.v1, w0.v1, w1.v1)
        wL.v2 = thinc_ul(wm1.v2, w0.v2, w1.v2)
    wL.v4 = thinc_ul(wm1.v4, w0.v4, w1.v4)
    if use_thinc:
        wR.v1 = thinc_ur(w0.v1, w1.v1, w2.v1)
        wR.v2 = thinc_ur(w0.v2, w1.v2, w2.v2)
    wR.v4 = thinc_ur(w0.v4, w1.v4, w2.v4)

    # use_thinc_L = (w1.v4 - w0.v4) * (w0.v4 - wm1.v4) > wp.float64(0.0) \
    #            and w0.v4 > WTHINC_THRESH \
    #            and w0.v4 < wp.float64(1.0) - WTHINC_THRESH
    
    # use_thinc_R = (w2.v4 - w1.v4) * (w1.v4 - w0.v4) > wp.float64(0.0) \
    #            and w1.v4 > WTHINC_THRESH \
    #            and w1.v4 < wp.float64(1.0) - WTHINC_THRESH
    
    # if use_thinc:
    #     wL.v1 = thinc_ul(wm1.v1, w0.v1, w1.v1)
    #     wL.v2 = thinc_ul(wm1.v2, w0.v2, w1.v2)
    #     wL.v4 = thinc_ul(wm1.v4, w0.v4, w1.v4)
    
    # if use_thinc:
    #     wR.v1 = thinc_ur(w0.v1, w1.v1, w2.v1)
    #     wR.v2 = thinc_ur(w0.v2, w1.v2, w2.v2)
    #     wR.v4 = thinc_ur(w0.v4, w1.v4, w2.v4)

    # ── Back-project to SC physical space ─────────────────────────────────
    phL = to_phys_x(wL, ca, rhoa, ar1a, ar2a, u_avg, v_avg)
    phR = to_phys_x(wR, ca, rhoa, ar1a, ar2a, u_avg, v_avg)

    # ── First-order fallback (zeroth-order char reconstruction) ───────────────
    phfL = to_phys_x(w0, ca, rhoa, ar1a, ar2a, u_avg, v_avg)
    phfR = to_phys_x(w1, ca, rhoa, ar1a, ar2a, u_avg, v_avg)

    phL = pp_limit_reconstructed_state(phL, phfL)
    phR = pp_limit_reconstructed_state(phR, phfR)

    # ── Positivity fix (matches Fortran) ──────────────────────────────────────
    if phL.v0 < wp.float64(0.0) or wp.isnan(phL.v0): phL.v0 = phfL.v0
    if phL.v1 < wp.float64(0.0) or wp.isnan(phL.v1): phL.v1 = phfL.v1
    if phL.v5 < wp.float64(0.0) or phL.v5 > W1:      phL.v5 = phfL.v5
    if rho_ctilde2_sc(phL) < WPP_CTILDE or wp.isnan(phL.v4):   phL.v4 = phfL.v4
    phL = pp_floor_pressure_state(phL)

    if phR.v0 < wp.float64(0.0) or wp.isnan(phR.v0): phR.v0 = phfR.v0
    if phR.v1 < wp.float64(0.0) or wp.isnan(phR.v1): phR.v1 = phfR.v1
    if phR.v5 < wp.float64(0.0) or phR.v5 > W1:      phR.v5 = phfR.v5
    if rho_ctilde2_sc(phR) < WPP_CTILDE or wp.isnan(phR.v4):   phR.v4 = phfR.v4
    phR = pp_floor_pressure_state(phR)

    if ca < wp.float64(0.0) or wp.isnan(ca):
        phL = phfL
        phR = phfR

    # ── Extract primitives for HLLC ───────────────────────────────────────────
    rhoL = phL.v0 + phL.v1;  rhoR = phR.v0 + phR.v1
    pUL0 = phL.v0; pUL1 = phL.v1
    pUL2 = phL.v2 / rhoL; pUL3 = phL.v3 / rhoL
    pUL4 = phL.v4; pUL5 = phL.v5

    pUR0 = phR.v0; pUR1 = phR.v1
    pUR2 = phR.v2 / rhoR; pUR3 = phR.v3 / rhoR
    pUR4 = phR.v4; pUR5 = phR.v5

    # ── HLLC ──────────────────────────────────────────────────────────────────
    fl = hllc_x(pUL0,pUL1,pUL2,pUL3,pUL4,pUL5, pUR0,pUR1,pUR2,pUR3,pUR4,pUR5)
    sx = hllc_x_src(pUL0,pUL1,pUL2,pUL4,pUL5,  pUR0,pUR1,pUR2,pUR4,pUR5)

    fx0[ii, jj] = fl.v0; fx1[ii, jj] = fl.v1; fx2[ii, jj] = fl.v2
    fx3[ii, jj] = fl.v3; fx4[ii, jj] = fl.v4; fx5[ii, jj] = fl.v5
    srcx[ii, jj] = sx


# ══════════════════════════════════════════════════════════════════════════════
#  gy_kernel  (same logic, y-direction)
# ══════════════════════════════════════════════════════════════════════════════

@wp.kernel
def gy_kernel(ar1: wp.array2d(dtype=wp.float64),
              ar2: wp.array2d(dtype=wp.float64),
              ru:  wp.array2d(dtype=wp.float64),
              rv:  wp.array2d(dtype=wp.float64),
              p_p: wp.array2d(dtype=wp.float64),
              al:  wp.array2d(dtype=wp.float64),
              gy0: wp.array2d(dtype=wp.float64),
              gy1: wp.array2d(dtype=wp.float64),
              gy2: wp.array2d(dtype=wp.float64),
              gy3: wp.array2d(dtype=wp.float64),
              gy4: wp.array2d(dtype=wp.float64),
              gy5: wp.array2d(dtype=wp.float64),
              srcy: wp.array2d(dtype=wp.float64),
              ofs_i: int, ofs_j: int):

    ii_loc, jj_loc = wp.tid()
    ii = ii_loc + ofs_i
    jj = jj_loc + ofs_j

    # ── Load stencil in y ─────────────────────────────────────────────────
    pm2 = S6()
    pm2.v0=ar1[ii,jj-2]; pm2.v1=ar2[ii,jj-2]
    pm2.v2=ru[ii,jj-2];  pm2.v3=rv[ii,jj-2]
    pm2.v4=p_p[ii,jj-2]; pm2.v5=al[ii,jj-2]

    pm1 = S6()
    pm1.v0=ar1[ii,jj-1]; pm1.v1=ar2[ii,jj-1]
    pm1.v2=ru[ii,jj-1];  pm1.v3=rv[ii,jj-1]
    pm1.v4=p_p[ii,jj-1]; pm1.v5=al[ii,jj-1]

    p0 = S6()
    p0.v0=ar1[ii,jj]; p0.v1=ar2[ii,jj]
    p0.v2=ru[ii,jj];  p0.v3=rv[ii,jj]
    p0.v4=p_p[ii,jj]; p0.v5=al[ii,jj]

    pp1 = S6()
    pp1.v0=ar1[ii,jj+1]; pp1.v1=ar2[ii,jj+1]
    pp1.v2=ru[ii,jj+1];  pp1.v3=rv[ii,jj+1]
    pp1.v4=p_p[ii,jj+1]; pp1.v5=al[ii,jj+1]

    pp2 = S6()
    pp2.v0=ar1[ii,jj+2]; pp2.v1=ar2[ii,jj+2]
    pp2.v2=ru[ii,jj+2];  pp2.v3=rv[ii,jj+2]
    pp2.v4=p_p[ii,jj+2]; pp2.v5=al[ii,jj+2]

    pp3 = S6()
    pp3.v0=ar1[ii,jj+3]; pp3.v1=ar2[ii,jj+3]
    pp3.v2=ru[ii,jj+3];  pp3.v3=rv[ii,jj+3]
    pp3.v4=p_p[ii,jj+3]; pp3.v5=al[ii,jj+3]

    e_m3 = cell_entropy(ar1[ii,jj-3], ar2[ii,jj-3], p_p[ii,jj-3], al[ii,jj-3])
    e_m2 = cell_entropy(ar1[ii,jj-2], ar2[ii,jj-2], p_p[ii,jj-2], al[ii,jj-2])
    e_m1 = cell_entropy(ar1[ii,jj-1], ar2[ii,jj-1], p_p[ii,jj-1], al[ii,jj-1])
    e_0  = cell_entropy(ar1[ii,jj+0], ar2[ii,jj+0], p_p[ii,jj+0], al[ii,jj+0])
    e_p1 = cell_entropy(ar1[ii,jj+1], ar2[ii,jj+1], p_p[ii,jj+1], al[ii,jj+1])
    e_p2 = cell_entropy(ar1[ii,jj+2], ar2[ii,jj+2], p_p[ii,jj+2], al[ii,jj+2])
    e_p3 = cell_entropy(ar1[ii,jj+3], ar2[ii,jj+3], p_p[ii,jj+3], al[ii,jj+3])
    e_p4 = cell_entropy(ar1[ii,jj+4], ar2[ii,jj+4], p_p[ii,jj+4], al[ii,jj+4])

    c_m1 = cator2_val(e_m3, e_m2, e_m1, e_0,  e_p1)
    c_0  = cator2_val(e_m2, e_m1, e_0,  e_p1, e_p2)
    c_p1 = cator2_val(e_m1, e_0,  e_p1, e_p2, e_p3)
    c_p2 = cator2_val(e_0,  e_p1, e_p2, e_p3, e_p4)
    use_thinc = wp.min(wp.min(c_m1, c_0), wp.min(c_p1, c_p2)) < WTHINC_THRESH

    # ── Interface-averaged Roe state ───────────────────────────────────────
    ar1a = W0p5 * (p0.v0 + pp1.v0)
    ar2a = W0p5 * (p0.v1 + pp1.v1)
    a1a  = W0p5 * (p0.v5 + pp1.v5)
    pa   = W0p5 * (p0.v4 + pp1.v4)
    rhoa = ar1a + ar2a
    u_avg = W0p5 * (p0.v2 + pp1.v2) / rhoa   
    v_avg = W0p5 * (p0.v3 + pp1.v3) / rhoa   
    ca    = sound_speed(rhoa, pa, a1a)
    gdum, pinf_a = mix_gamma_pinf(a1a)

    # ── Always characteristic reconstruction ───────────────────────────────
    wm2 = to_char_y(pm2.v0,pm2.v1,pm2.v2,pm2.v3,pm2.v4,pm2.v5, ca,rhoa,ar1a,ar2a, u_avg,v_avg)
    wm1 = to_char_y(pm1.v0,pm1.v1,pm1.v2,pm1.v3,pm1.v4,pm1.v5, ca,rhoa,ar1a,ar2a, u_avg,v_avg)
    w0  = to_char_y(p0.v0, p0.v1, p0.v2, p0.v3, p0.v4, p0.v5,  ca,rhoa,ar1a,ar2a, u_avg,v_avg)
    w1  = to_char_y(pp1.v0,pp1.v1,pp1.v2,pp1.v3,pp1.v4,pp1.v5, ca,rhoa,ar1a,ar2a, u_avg,v_avg)
    w2  = to_char_y(pp2.v0,pp2.v1,pp2.v2,pp2.v3,pp2.v4,pp2.v5, ca,rhoa,ar1a,ar2a, u_avg,v_avg)
    w3  = to_char_y(pp3.v0,pp3.v1,pp3.v2,pp3.v3,pp3.v4,pp3.v5, ca,rhoa,ar1a,ar2a, u_avg,v_avg)

    wL = S6(); wR = S6()

    wL.v0 = muscl_char_ul(wm1.v0, w0.v0, w1.v0, w2.v0)
    wL.v1 = muscl_char_ul(wm1.v1, w0.v1, w1.v1, w2.v1)
    wL.v2 = muscl_char_ul(wm1.v2, w0.v2, w1.v2, w2.v2)
    wL.v3 = muscl_char_ul(wm1.v3, w0.v3, w1.v3, w2.v3)
    wL.v4 = muscl_char_ul(wm1.v4, w0.v4, w1.v4, w2.v4)
    wL.v5 = muscl_char_ul(wm1.v5, w0.v5, w1.v5, w2.v5)
    wR.v0 = muscl_char_ur(wm1.v0, w0.v0, w1.v0, w2.v0)
    wR.v1 = muscl_char_ur(wm1.v1, w0.v1, w1.v1, w2.v1)
    wR.v2 = muscl_char_ur(wm1.v2, w0.v2, w1.v2, w2.v2)
    wR.v3 = muscl_char_ur(wm1.v3, w0.v3, w1.v3, w2.v3)
    wR.v4 = muscl_char_ur(wm1.v4, w0.v4, w1.v4, w2.v4)
    wR.v5 = muscl_char_ur(wm1.v5, w0.v5, w1.v5, w2.v5)
    # # THINC override for v1, v2, v4 (Fortran i=2,3,5)
    if use_thinc:
        wL.v1 = thinc_ul(wm1.v1, w0.v1, w1.v1)
        wL.v2 = thinc_ul(wm1.v2, w0.v2, w1.v2)
    wL.v4 = thinc_ul(wm1.v4, w0.v4, w1.v4)
    if use_thinc:
        wR.v1 = thinc_ur(w0.v1, w1.v1, w2.v1)
        wR.v2 = thinc_ur(w0.v2, w1.v2, w2.v2)
    wR.v4 = thinc_ur(w0.v4, w1.v4, w2.v4)

    # # Simple criterio
    # use_thinc_L = (w1.v4 - w0.v4) * (w0.v4 - wm1.v4) > wp.float64(0.0) \
    #            and w0.v4 > WTHINC_THRESH \
    #            and w0.v4 < wp.float64(1.0) - WTHINC_THRESH
    
    # use_thinc_R = (w2.v4 - w1.v4) * (w1.v4 - w0.v4) > wp.float64(0.0) \
    #            and w1.v4 > WTHINC_THRESH \
    #            and w1.v4 < wp.float64(1.0) - WTHINC_THRESH
    
    # if use_thinc:
    #     wL.v1 = thinc_ul(wm1.v1, w0.v1, w1.v1)
    #     wL.v2 = thinc_ul(wm1.v2, w0.v2, w1.v2)
    #     wL.v4 = thinc_ul(wm1.v4, w0.v4, w1.v4)
    
    # if use_thinc:
    #     wR.v1 = thinc_ur(w0.v1, w1.v1, w2.v1)
    #     wR.v2 = thinc_ur(w0.v2, w1.v2, w2.v2)
    #     wR.v4 = thinc_ur(w0.v4, w1.v4, w2.v4)

    # ── Back-project ───────────────────────────────────────────────────────
    phL = to_phys_y(wL, ca, rhoa, ar1a, ar2a, u_avg, v_avg)
    phR = to_phys_y(wR, ca, rhoa, ar1a, ar2a, u_avg, v_avg)

    # ── First-order fallback ───────────────────────────────────────────────────
    phfL = to_phys_y(w0, ca, rhoa, ar1a, ar2a, u_avg, v_avg)
    phfR = to_phys_y(w1, ca, rhoa, ar1a, ar2a, u_avg, v_avg)

    phL = pp_limit_reconstructed_state(phL, phfL)
    phR = pp_limit_reconstructed_state(phR, phfR)

    # ── Positivity fix ─────────────────────────────────────────────────────────
    if phL.v0 < wp.float64(0.0) or wp.isnan(phL.v0): phL.v0 = phfL.v0
    if phL.v1 < wp.float64(0.0) or wp.isnan(phL.v1): phL.v1 = phfL.v1
    if phL.v5 < wp.float64(0.0) or phL.v5 > W1:      phL.v5 = phfL.v5
    if rho_ctilde2_sc(phL) < WPP_CTILDE or wp.isnan(phL.v4):   phL.v4 = phfL.v4
    phL = pp_floor_pressure_state(phL)

    if phR.v0 < wp.float64(0.0) or wp.isnan(phR.v0): phR.v0 = phfR.v0
    if phR.v1 < wp.float64(0.0) or wp.isnan(phR.v1): phR.v1 = phfR.v1
    if phR.v5 < wp.float64(0.0) or phR.v5 > W1:      phR.v5 = phfR.v5
    if rho_ctilde2_sc(phR) < WPP_CTILDE or wp.isnan(phR.v4):   phR.v4 = phfR.v4
    phR = pp_floor_pressure_state(phR)

    if ca < wp.float64(0.0) or wp.isnan(ca):
        phL = phfL
        phR = phfR

    # ── Extract primitives for HLLC ───────────────────────────────────────────
    rhoL = phL.v0 + phL.v1;  rhoR = phR.v0 + phR.v1
    pUL0 = phL.v0; pUL1 = phL.v1
    pUL2 = phL.v2 / rhoL; pUL3 = phL.v3 / rhoL
    pUL4 = phL.v4; pUL5 = phL.v5

    pUR0 = phR.v0; pUR1 = phR.v1
    pUR2 = phR.v2 / rhoR; pUR3 = phR.v3 / rhoR
    pUR4 = phR.v4; pUR5 = phR.v5

    # ── HLLC ──────────────────────────────────────────────────────────────────
    fl = hllc_y(pUL0,pUL1,pUL2,pUL3,pUL4,pUL5, pUR0,pUR1,pUR2,pUR3,pUR4,pUR5)
    sy = hllc_y_src(pUL0,pUL1,pUL3,pUL4,pUL5,  pUR0,pUR1,pUR3,pUR4,pUR5)

    gy0[ii, jj] = fl.v0; gy1[ii, jj] = fl.v1; gy2[ii, jj] = fl.v2
    gy3[ii, jj] = fl.v3; gy4[ii, jj] = fl.v4; gy5[ii, jj] = fl.v5
    srcy[ii, jj] = sy


@wp.kernel
def fx_low_kernel(ar1: wp.array2d(dtype=wp.float64),
                  ar2: wp.array2d(dtype=wp.float64),
                  ru:  wp.array2d(dtype=wp.float64),
                  rv:  wp.array2d(dtype=wp.float64),
                  p_p: wp.array2d(dtype=wp.float64),
                  al:  wp.array2d(dtype=wp.float64),
                  fx0: wp.array2d(dtype=wp.float64),
                  fx1: wp.array2d(dtype=wp.float64),
                  fx2: wp.array2d(dtype=wp.float64),
                  fx3: wp.array2d(dtype=wp.float64),
                  fx4: wp.array2d(dtype=wp.float64),
                  fx5: wp.array2d(dtype=wp.float64),
                  srcx: wp.array2d(dtype=wp.float64),
                  ofs_i: int, ofs_j: int):
    ii_loc, jj_loc = wp.tid()
    ii = ii_loc + ofs_i
    jj = jj_loc + ofs_j

    rhoL = ar1[ii, jj] + ar2[ii, jj]
    rhoR = ar1[ii+1, jj] + ar2[ii+1, jj]
    uL = ru[ii, jj] / rhoL
    vL = rv[ii, jj] / rhoL
    uR = ru[ii+1, jj] / rhoR
    vR = rv[ii+1, jj] / rhoR

    aL = clamp01_pp(al[ii, jj])
    aR = clamp01_pp(al[ii+1, jj])
    pL = p_p[ii, jj]
    pR = p_p[ii+1, jj]
    gL, piL = mix_gamma_pinf(aL)
    gR, piR = mix_gamma_pinf(aR)
    if wp.isnan(pL) or (pL + piL) / (gL - W1) < WPP_CTILDE:
        pL = WPP_CTILDE * (gL - W1) - piL
    if wp.isnan(pR) or (pR + piR) / (gR - W1) < WPP_CTILDE:
        pR = WPP_CTILDE * (gR - W1) - piR

    fl = hllc_x(ar1[ii,jj], ar2[ii,jj], uL, vL, pL, aL,
                ar1[ii+1,jj], ar2[ii+1,jj], uR, vR, pR, aR)
    sx = hllc_x_src(ar1[ii,jj], ar2[ii,jj], uL, pL, aL,
                    ar1[ii+1,jj], ar2[ii+1,jj], uR, pR, aR)

    fx0[ii, jj] = fl.v0; fx1[ii, jj] = fl.v1; fx2[ii, jj] = fl.v2
    fx3[ii, jj] = fl.v3; fx4[ii, jj] = fl.v4; fx5[ii, jj] = fl.v5
    srcx[ii, jj] = sx


@wp.kernel
def gy_low_kernel(ar1: wp.array2d(dtype=wp.float64),
                  ar2: wp.array2d(dtype=wp.float64),
                  ru:  wp.array2d(dtype=wp.float64),
                  rv:  wp.array2d(dtype=wp.float64),
                  p_p: wp.array2d(dtype=wp.float64),
                  al:  wp.array2d(dtype=wp.float64),
                  gy0: wp.array2d(dtype=wp.float64),
                  gy1: wp.array2d(dtype=wp.float64),
                  gy2: wp.array2d(dtype=wp.float64),
                  gy3: wp.array2d(dtype=wp.float64),
                  gy4: wp.array2d(dtype=wp.float64),
                  gy5: wp.array2d(dtype=wp.float64),
                  srcy: wp.array2d(dtype=wp.float64),
                  ofs_i: int, ofs_j: int):
    ii_loc, jj_loc = wp.tid()
    ii = ii_loc + ofs_i
    jj = jj_loc + ofs_j

    rhoL = ar1[ii, jj] + ar2[ii, jj]
    rhoR = ar1[ii, jj+1] + ar2[ii, jj+1]
    uL = ru[ii, jj] / rhoL
    vL = rv[ii, jj] / rhoL
    uR = ru[ii, jj+1] / rhoR
    vR = rv[ii, jj+1] / rhoR

    aL = clamp01_pp(al[ii, jj])
    aR = clamp01_pp(al[ii, jj+1])
    pL = p_p[ii, jj]
    pR = p_p[ii, jj+1]
    gL, piL = mix_gamma_pinf(aL)
    gR, piR = mix_gamma_pinf(aR)
    if wp.isnan(pL) or (pL + piL) / (gL - W1) < WPP_CTILDE:
        pL = WPP_CTILDE * (gL - W1) - piL
    if wp.isnan(pR) or (pR + piR) / (gR - W1) < WPP_CTILDE:
        pR = WPP_CTILDE * (gR - W1) - piR

    fl = hllc_y(ar1[ii,jj], ar2[ii,jj], uL, vL, pL, aL,
                ar1[ii,jj+1], ar2[ii,jj+1], uR, vR, pR, aR)
    sy = hllc_y_src(ar1[ii,jj], ar2[ii,jj], vL, pL, aL,
                    ar1[ii,jj+1], ar2[ii,jj+1], vR, pR, aR)

    gy0[ii, jj] = fl.v0; gy1[ii, jj] = fl.v1; gy2[ii, jj] = fl.v2
    gy3[ii, jj] = fl.v3; gy4[ii, jj] = fl.v4; gy5[ii, jj] = fl.v5
    srcy[ii, jj] = sy


@wp.kernel
def pp_cell_theta_kernel(ar1: wp.array2d(dtype=wp.float64),
                         ar2: wp.array2d(dtype=wp.float64),
                         ru:  wp.array2d(dtype=wp.float64),
                         rv:  wp.array2d(dtype=wp.float64),
                         E:   wp.array2d(dtype=wp.float64),
                         al:  wp.array2d(dtype=wp.float64),
                         al_prim: wp.array2d(dtype=wp.float64),
                         fx0: wp.array2d(dtype=wp.float64),
                         fx1: wp.array2d(dtype=wp.float64),
                         fx2: wp.array2d(dtype=wp.float64),
                         fx3: wp.array2d(dtype=wp.float64),
                         fx4: wp.array2d(dtype=wp.float64),
                         fx5: wp.array2d(dtype=wp.float64),
                         srcx: wp.array2d(dtype=wp.float64),
                         gy0: wp.array2d(dtype=wp.float64),
                         gy1: wp.array2d(dtype=wp.float64),
                         gy2: wp.array2d(dtype=wp.float64),
                         gy3: wp.array2d(dtype=wp.float64),
                         gy4: wp.array2d(dtype=wp.float64),
                         gy5: wp.array2d(dtype=wp.float64),
                         srcy: wp.array2d(dtype=wp.float64),
                         fxl0: wp.array2d(dtype=wp.float64),
                         fxl1: wp.array2d(dtype=wp.float64),
                         fxl2: wp.array2d(dtype=wp.float64),
                         fxl3: wp.array2d(dtype=wp.float64),
                         fxl4: wp.array2d(dtype=wp.float64),
                         fxl5: wp.array2d(dtype=wp.float64),
                         srcxl: wp.array2d(dtype=wp.float64),
                         gyl0: wp.array2d(dtype=wp.float64),
                         gyl1: wp.array2d(dtype=wp.float64),
                         gyl2: wp.array2d(dtype=wp.float64),
                         gyl3: wp.array2d(dtype=wp.float64),
                         gyl4: wp.array2d(dtype=wp.float64),
                         gyl5: wp.array2d(dtype=wp.float64),
                         srcyl: wp.array2d(dtype=wp.float64),
                         theta_cell: wp.array2d(dtype=wp.float64),
                         dx: wp.float64, dy: wp.float64, dt: wp.float64,
                         ofs_i: int, ofs_j: int, stage: int):
    ii_loc, jj_loc = wp.tid()
    i = ii_loc + ofs_i
    j = jj_loc + ofs_j

    inv_dx = W1 / dx
    inv_dy = W1 / dy
    a = al_prim[i, j]

    rh0 = (fx0[i,j] - fx0[i-1,j]) * inv_dx + (gy0[i,j] - gy0[i,j-1]) * inv_dy
    rh1 = (fx1[i,j] - fx1[i-1,j]) * inv_dx + (gy1[i,j] - gy1[i,j-1]) * inv_dy
    rh2 = (fx2[i,j] - fx2[i-1,j]) * inv_dx + (gy2[i,j] - gy2[i,j-1]) * inv_dy
    rh3 = (fx3[i,j] - fx3[i-1,j]) * inv_dx + (gy3[i,j] - gy3[i,j-1]) * inv_dy
    rh4 = (fx4[i,j] - fx4[i-1,j]) * inv_dx + (gy4[i,j] - gy4[i,j-1]) * inv_dy
    rh5 = (fx5[i,j] - fx5[i-1,j]) * inv_dx + (gy5[i,j] - gy5[i,j-1]) * inv_dy \
        - a * (srcx[i,j] - srcx[i-1,j]) * inv_dx \
        - a * (srcy[i,j] - srcy[i,j-1]) * inv_dy

    rl0 = (fxl0[i,j] - fxl0[i-1,j]) * inv_dx + (gyl0[i,j] - gyl0[i,j-1]) * inv_dy
    rl1 = (fxl1[i,j] - fxl1[i-1,j]) * inv_dx + (gyl1[i,j] - gyl1[i,j-1]) * inv_dy
    rl2 = (fxl2[i,j] - fxl2[i-1,j]) * inv_dx + (gyl2[i,j] - gyl2[i,j-1]) * inv_dy
    rl3 = (fxl3[i,j] - fxl3[i-1,j]) * inv_dx + (gyl3[i,j] - gyl3[i,j-1]) * inv_dy
    rl4 = (fxl4[i,j] - fxl4[i-1,j]) * inv_dx + (gyl4[i,j] - gyl4[i,j-1]) * inv_dy
    rl5 = (fxl5[i,j] - fxl5[i-1,j]) * inv_dx + (gyl5[i,j] - gyl5[i,j-1]) * inv_dy \
        - a * (srcxl[i,j] - srcxl[i-1,j]) * inv_dx \
        - a * (srcyl[i,j] - srcyl[i,j-1]) * inv_dy

    l0 = ar1[i,j] - dt * rl0
    l1 = ar2[i,j] - dt * rl1
    l2 = ru[i,j]  - dt * rl2
    l3 = rv[i,j]  - dt * rl3
    l4 = E[i,j]   - dt * rl4
    l5 = al[i,j]  - dt * rl5

    h0 = ar1[i,j] - dt * rh0
    h1 = ar2[i,j] - dt * rh1
    h2 = ru[i,j]  - dt * rh2
    h3 = rv[i,j]  - dt * rh3
    h4 = E[i,j]   - dt * rh4
    h5 = al[i,j]  - dt * rh5

    theta_cell[i, j] = theta_cons_stage(l0,l1,l2,l3,l4,l5, h0,h1,h2,h3,h4,h5, stage)


@wp.kernel
def pp_blend_x_flux_kernel(fx0: wp.array2d(dtype=wp.float64),
                           fx1: wp.array2d(dtype=wp.float64),
                           fx2: wp.array2d(dtype=wp.float64),
                           fx3: wp.array2d(dtype=wp.float64),
                           fx4: wp.array2d(dtype=wp.float64),
                           fx5: wp.array2d(dtype=wp.float64),
                           srcx: wp.array2d(dtype=wp.float64),
                           fxl0: wp.array2d(dtype=wp.float64),
                           fxl1: wp.array2d(dtype=wp.float64),
                           fxl2: wp.array2d(dtype=wp.float64),
                           fxl3: wp.array2d(dtype=wp.float64),
                           fxl4: wp.array2d(dtype=wp.float64),
                           fxl5: wp.array2d(dtype=wp.float64),
                           srcxl: wp.array2d(dtype=wp.float64),
                           theta_cell: wp.array2d(dtype=wp.float64),
                           ofs_i: int, ofs_j: int, g_: int, nx_: int):
    ii_loc, jj_loc = wp.tid()
    i = ii_loc + ofs_i
    j = jj_loc + ofs_j

    theta = W1
    if i >= g_ and i < g_ + nx_:
        theta = wp.min(theta, theta_cell[i, j])
    if i + 1 >= g_ and i + 1 < g_ + nx_:
        theta = wp.min(theta, theta_cell[i+1, j])

    fx0[i,j] = fxl0[i,j] + theta * (fx0[i,j] - fxl0[i,j])
    fx1[i,j] = fxl1[i,j] + theta * (fx1[i,j] - fxl1[i,j])
    fx2[i,j] = fxl2[i,j] + theta * (fx2[i,j] - fxl2[i,j])
    fx3[i,j] = fxl3[i,j] + theta * (fx3[i,j] - fxl3[i,j])
    fx4[i,j] = fxl4[i,j] + theta * (fx4[i,j] - fxl4[i,j])
    fx5[i,j] = fxl5[i,j] + theta * (fx5[i,j] - fxl5[i,j])
    srcx[i,j] = srcxl[i,j] + theta * (srcx[i,j] - srcxl[i,j])


@wp.kernel
def pp_blend_y_flux_kernel(gy0: wp.array2d(dtype=wp.float64),
                           gy1: wp.array2d(dtype=wp.float64),
                           gy2: wp.array2d(dtype=wp.float64),
                           gy3: wp.array2d(dtype=wp.float64),
                           gy4: wp.array2d(dtype=wp.float64),
                           gy5: wp.array2d(dtype=wp.float64),
                           srcy: wp.array2d(dtype=wp.float64),
                           gyl0: wp.array2d(dtype=wp.float64),
                           gyl1: wp.array2d(dtype=wp.float64),
                           gyl2: wp.array2d(dtype=wp.float64),
                           gyl3: wp.array2d(dtype=wp.float64),
                           gyl4: wp.array2d(dtype=wp.float64),
                           gyl5: wp.array2d(dtype=wp.float64),
                           srcyl: wp.array2d(dtype=wp.float64),
                           theta_cell: wp.array2d(dtype=wp.float64),
                           ofs_i: int, ofs_j: int, g_: int, ny_: int):
    ii_loc, jj_loc = wp.tid()
    i = ii_loc + ofs_i
    j = jj_loc + ofs_j

    theta = W1
    if j >= g_ and j < g_ + ny_:
        theta = wp.min(theta, theta_cell[i, j])
    if j + 1 >= g_ and j + 1 < g_ + ny_:
        theta = wp.min(theta, theta_cell[i, j+1])

    gy0[i,j] = gyl0[i,j] + theta * (gy0[i,j] - gyl0[i,j])
    gy1[i,j] = gyl1[i,j] + theta * (gy1[i,j] - gyl1[i,j])
    gy2[i,j] = gyl2[i,j] + theta * (gy2[i,j] - gyl2[i,j])
    gy3[i,j] = gyl3[i,j] + theta * (gy3[i,j] - gyl3[i,j])
    gy4[i,j] = gyl4[i,j] + theta * (gy4[i,j] - gyl4[i,j])
    gy5[i,j] = gyl5[i,j] + theta * (gy5[i,j] - gyl5[i,j])
    srcy[i,j] = srcyl[i,j] + theta * (srcy[i,j] - srcyl[i,j])



# ── RK3 update kernel ─────────────────────────────────────────────────────────
@wp.kernel
def rk3_kernel(# old state
               ar1_old: wp.array2d(dtype=wp.float64),
               ar2_old: wp.array2d(dtype=wp.float64),
               ru_old:  wp.array2d(dtype=wp.float64),
               rv_old:  wp.array2d(dtype=wp.float64),
               E_old:   wp.array2d(dtype=wp.float64),
               al_old:  wp.array2d(dtype=wp.float64),
               # current (stage) state
               ar1: wp.array2d(dtype=wp.float64),
               ar2: wp.array2d(dtype=wp.float64),
               ru:  wp.array2d(dtype=wp.float64),
               rv:  wp.array2d(dtype=wp.float64),
               E:   wp.array2d(dtype=wp.float64),
               al:  wp.array2d(dtype=wp.float64),
               al_prim: wp.array2d(dtype=wp.float64),  # α₁ primitive (for source)
               # fluxes
               fx0: wp.array2d(dtype=wp.float64),
               fx1: wp.array2d(dtype=wp.float64),
               fx2: wp.array2d(dtype=wp.float64),
               fx3: wp.array2d(dtype=wp.float64),
               fx4: wp.array2d(dtype=wp.float64),
               fx5: wp.array2d(dtype=wp.float64),
               srcx: wp.array2d(dtype=wp.float64),
               gy0: wp.array2d(dtype=wp.float64),
               gy1: wp.array2d(dtype=wp.float64),
               gy2: wp.array2d(dtype=wp.float64),
               gy3: wp.array2d(dtype=wp.float64),
               gy4: wp.array2d(dtype=wp.float64),
               gy5: wp.array2d(dtype=wp.float64),
               srcy: wp.array2d(dtype=wp.float64),
               dx: wp.float64, dy: wp.float64,
               dt: wp.float64,
               alpha: wp.float64, beta: wp.float64,
               ofs_i: int, ofs_j: int):
    ii_loc, jj_loc = wp.tid()
    i = ii_loc + ofs_i
    j = jj_loc + ofs_j

    inv_dx = W1 / dx;  inv_dy = W1 / dy

    res0 = (fx0[i, j] - fx0[i-1, j]) * inv_dx + (gy0[i, j] - gy0[i, j-1]) * inv_dy
    res1 = (fx1[i, j] - fx1[i-1, j]) * inv_dx + (gy1[i, j] - gy1[i, j-1]) * inv_dy
    res2 = (fx2[i, j] - fx2[i-1, j]) * inv_dx + (gy2[i, j] - gy2[i, j-1]) * inv_dy
    res3 = (fx3[i, j] - fx3[i-1, j]) * inv_dx + (gy3[i, j] - gy3[i, j-1]) * inv_dy
    res4 = (fx4[i, j] - fx4[i-1, j]) * inv_dx + (gy4[i, j] - gy4[i, j-1]) * inv_dy

    # Non-conservative source for α₁ equation
    src6x = al_prim[i, j] * (srcx[i, j] - srcx[i-1, j]) * inv_dx
    src6y = al_prim[i, j] * (srcy[i, j] - srcy[i, j-1]) * inv_dy
    res5 = (fx5[i, j] - fx5[i-1, j]) * inv_dx + (gy5[i, j] - gy5[i, j-1]) * inv_dy \
           - src6x - src6y

    ar1[i, j] = alpha*ar1_old[i, j] + beta*(ar1[i, j] - dt*res0)
    ar2[i, j] = alpha*ar2_old[i, j] + beta*(ar2[i, j] - dt*res1)
    ru[i, j]  = alpha*ru_old[i, j]  + beta*(ru[i, j]  - dt*res2)
    rv[i, j]  = alpha*rv_old[i, j]  + beta*(rv[i, j]  - dt*res3)
    E[i, j]   = alpha*E_old[i, j]   + beta*(E[i, j]   - dt*res4)
    al[i, j]  = alpha*al_old[i, j]  + beta*(al[i, j]  - dt*res5)


# ── Boundary conditions ───────────────────────────────────────────────────────
# x: transmissive (copy nearest interior cell into ghost)
@wp.kernel
def bc_x_kernel(ar1: wp.array2d(dtype=wp.float64),
                ar2: wp.array2d(dtype=wp.float64),
                ru:  wp.array2d(dtype=wp.float64),
                rv:  wp.array2d(dtype=wp.float64),
                E:   wp.array2d(dtype=wp.float64),
                al:  wp.array2d(dtype=wp.float64),
                u_p: wp.array2d(dtype=wp.float64),
                v_p: wp.array2d(dtype=wp.float64),
                p_p: wp.array2d(dtype=wp.float64),
                g_: int, nx_: int, ny_: int):
    g_loc, j = wp.tid()   # g_loc in [0, G-1], j in [0, NYG-1]
    k = g_loc              # ghost layer index (0=outermost)

    # Left ghost: copies cell G (interior left boundary)
    li = g_ - 1 - k       # ghost index: G-1, G-2, ..., 0
    src_l = g_             # interior source: always cell G
    ar1[li, j] = ar1[src_l, j];  ar2[li, j] = ar2[src_l, j]
    ru[li, j]  = ru[src_l, j];   rv[li, j]  = rv[src_l, j]
    E[li, j]   = E[src_l, j];    al[li, j]  = al[src_l, j]
    u_p[li, j] = u_p[src_l, j];  v_p[li, j] = v_p[src_l, j]
    p_p[li, j] = p_p[src_l, j]

    # Right ghost
    ri = g_ + nx_ + k     # ghost index: G+NX, G+NX+1, ...
    src_r = g_ + nx_ - 1  # interior source: cell G+NX-1
    ar1[ri, j] = ar1[src_r, j];  ar2[ri, j] = ar2[src_r, j]
    ru[ri, j]  = ru[src_r, j];   rv[ri, j]  = rv[src_r, j]
    E[ri, j]   = E[src_r, j];    al[ri, j]  = al[src_r, j]
    u_p[ri, j] = u_p[src_r, j];  v_p[ri, j] = v_p[src_r, j]
    p_p[ri, j] = p_p[src_r, j]


# y: slip wall (reflect v, copy all else)
@wp.kernel
def bc_y_kernel(ar1: wp.array2d(dtype=wp.float64),
                ar2: wp.array2d(dtype=wp.float64),
                ru:  wp.array2d(dtype=wp.float64),
                rv:  wp.array2d(dtype=wp.float64),
                E:   wp.array2d(dtype=wp.float64),
                al:  wp.array2d(dtype=wp.float64),
                u_p: wp.array2d(dtype=wp.float64),
                v_p: wp.array2d(dtype=wp.float64),
                p_p: wp.array2d(dtype=wp.float64),
                g_: int, ny_: int):
    i, g_loc = wp.tid()    # i in [0, NXG-1], g_loc in [0, G-1]
    k = g_loc

    # Bottom ghost: reflect v
    bj = g_ - 1 - k       # ghost: G-1, G-2, ...
    src_b = g_             # interior source: cell G
    ar1[i, bj] = ar1[i, src_b];  ar2[i, bj] = ar2[i, src_b]
    ru[i, bj]  = ru[i, src_b]
    rv[i, bj]  = wp.float64(-1.0) * rv[i, src_b]   # reflect v
    E[i, bj]   = E[i, src_b];    al[i, bj]  = al[i, src_b]
    u_p[i, bj] = u_p[i, src_b];  v_p[i, bj] = wp.float64(-1.0)*v_p[i, src_b]
    p_p[i, bj] = p_p[i, src_b]

    # Top ghost: reflect v
    tj = g_ + ny_ + k
    src_t = g_ + ny_ - 1
    ar1[i, tj] = ar1[i, src_t];  ar2[i, tj] = ar2[i, src_t]
    ru[i, tj]  = ru[i, src_t]
    rv[i, tj]  = wp.float64(-1.0) * rv[i, src_t]
    E[i, tj]   = E[i, src_t];    al[i, tj]  = al[i, src_t]
    u_p[i, tj] = u_p[i, src_t];  v_p[i, tj] = wp.float64(-1.0)*v_p[i, src_t]
    p_p[i, tj] = p_p[i, src_t]


# ── Timestep reduction ────────────────────────────────────────────────────────
@wp.kernel
def timestep_kernel(ar1: wp.array2d(dtype=wp.float64),
                    ar2: wp.array2d(dtype=wp.float64),
                    u_p: wp.array2d(dtype=wp.float64),
                    v_p: wp.array2d(dtype=wp.float64),
                    p_p: wp.array2d(dtype=wp.float64),
                    al:  wp.array2d(dtype=wp.float64),
                    dx: wp.float64, dy: wp.float64,
                    dt_buf: wp.array(dtype=wp.float64),
                    ofs_i: int, ofs_j: int):
    ii_loc, jj_loc = wp.tid()
    i = ii_loc + ofs_i
    j = jj_loc + ofs_j

    rho = ar1[i, j] + ar2[i, j]
    c   = sound_speed(rho, p_p[i, j], al[i, j])
    sx  = dx / (wp.abs(u_p[i, j]) + c)
    sy  = dy / (wp.abs(v_p[i, j]) + c)
    dt_loc = sx * sy / (sx + sy)
    wp.atomic_min(dt_buf, 0, dt_loc)


# ══════════════════════════════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════════════════════════════

def save_npz(ar1_np, ar2_np, u_np, v_np, p_np, al_np, x_np, y_np, step, t):
    rho_np = ar1_np + ar2_np
    fname  = f"mach10_{step:06d}.npz"
    np.savez_compressed(fname,
                        time=np.float64(t),
                        x=x_np[G:G+NX], y=y_np[G:G+NY],
                        rho=rho_np[G:G+NX, G:G+NY],
                        ar1=ar1_np[G:G+NX, G:G+NY],
                        ar2=ar2_np[G:G+NX, G:G+NY],
                        u=u_np[G:G+NX, G:G+NY],
                        v=v_np[G:G+NX, G:G+NY],
                        p=p_np[G:G+NX, G:G+NY],
                        alpha1=al_np[G:G+NX, G:G+NY])
    print(f"  saved {fname}  t={t:.2e}")


def save_png(ar1_np, ar2_np, u_np, v_np, p_np, al_np, x_np, y_np, step, t):
    rho = (ar1_np + ar2_np)[G:G+NX, G:G+NY]   # shape (NX, NY)
    al  = al_np[G:G+NX, G:G+NY]
    xg  = x_np[G:G+NX]
    yg  = y_np[G:G+NY]


    drhodx = np.gradient(rho, xg, axis=0)
    drhody = np.gradient(rho, yg, axis=1)
    grad   = np.sqrt(drhodx**2 + drhody**2)   # shape (NX, NY)


    grad_T = grad.T                             # shape (NY, NX)


    plt.rcParams['axes.xmargin'] = 0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ── Schlieren (left panel) ─────────────────────────────────────────────
    ax1 = axes[0]
    ax1.imshow(grad_T, vmin=1, vmax=2400,
               cmap=plt.cm.Blues, origin='upper',
               extent=[xg.min(), xg.max(), yg.min(), yg.max()])
    ax1.contour(xg, yg, al.T, [0.5], colors='r', linewidths=0.5)
    # ax1.set_xlabel(r'\textbf{x}' if plt.rcParams.get('text.usetex') else 'x')
    # ax1.set_ylabel(r'\textbf{y}' if plt.rcParams.get('text.usetex') else 'y')
    ax1.axis('off')
    # ax1.set_title(rf'$t = {t*1e6:.1f}\ \mu$s')

    # ── Volume fraction α₁ (right panel) ──────────────────────────────────
    ax2 = axes[1]
    ax2.imshow(al.T, vmin=0, vmax=1,
               cmap=plt.cm.gray_r, origin='upper',
               extent=[xg.min(), xg.max(), yg.min(), yg.max()])
    ax2.axis('off')
    ax2.set_title(r'$\alpha_1$')

    fig.tight_layout(pad=0.3)
    fname = f"mach10_{step:06d}.png"
    fig.savefig(fname, dpi=300, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)
    print(f"  saved {fname}")

def infer_restart_step(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    tail = stem.split("_")[-1]
    return int(tail) if tail.isdigit() else 0


def load_restart_host(path, x_np, y_np):
    data = np.load(path)
    required = ["ar1", "ar2", "u", "v", "p", "alpha1"]
    for key in required:
        if key not in data:
            raise KeyError(f"Restart file {path} is missing '{key}'")

    ar1_i = np.asarray(data["ar1"], dtype=np.float64)
    ar2_i = np.asarray(data["ar2"], dtype=np.float64)
    u_i   = np.asarray(data["u"], dtype=np.float64)
    v_i   = np.asarray(data["v"], dtype=np.float64)
    p_i   = np.asarray(data["p"], dtype=np.float64)
    al_i  = np.asarray(data["alpha1"], dtype=np.float64)

    if ar1_i.shape != (NX, NY):
        raise ValueError(f"Restart shape {ar1_i.shape} does not match {(NX, NY)}")

    ar1_np = np.zeros((NXG, NYG), dtype=np.float64)
    ar2_np = np.zeros((NXG, NYG), dtype=np.float64)
    ru_np  = np.zeros((NXG, NYG), dtype=np.float64)
    rv_np  = np.zeros((NXG, NYG), dtype=np.float64)
    E_np   = np.zeros((NXG, NYG), dtype=np.float64)
    al_np  = np.zeros((NXG, NYG), dtype=np.float64)
    u_np   = np.zeros((NXG, NYG), dtype=np.float64)
    v_np   = np.zeros((NXG, NYG), dtype=np.float64)
    p_np   = np.zeros((NXG, NYG), dtype=np.float64)

    sl = np.s_[G:G+NX, G:G+NY]
    al_i = np.clip(al_i, 1.0e-8, 1.0 - 1.0e-8)
    rho_i = ar1_i + ar2_i
    ru_i = rho_i * u_i
    rv_i = rho_i * v_i
    gb = al_i / (6.12 - 1.0) + (1.0 - al_i) / (1.4 - 1.0)
    pb = al_i * 6.12 * 3.43e8 / (6.12 - 1.0)
    E_i = gb * p_i + pb + 0.5 * rho_i * (u_i*u_i + v_i*v_i)

    ar1_np[sl] = ar1_i; ar2_np[sl] = ar2_i
    ru_np[sl] = ru_i;   rv_np[sl] = rv_i
    E_np[sl] = E_i;     al_np[sl] = al_i
    u_np[sl] = u_i;     v_np[sl] = v_i; p_np[sl] = p_i

    restart_time = float(data["time"]) if "time" in data else 0.0
    return ar1_np, ar2_np, ru_np, rv_np, E_np, al_np, u_np, v_np, p_np, restart_time


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"Device: {DEVICE}  NX={NX}  NY={NY}  G={G}")

    # Grid coordinates
    x_np = np.array([XMIN_V + (i - G + 0.5)*DX_V for i in range(NXG)], dtype=np.float64)
    y_np = np.array([YMIN_V + (j - G + 0.5)*DY_V for j in range(NYG)], dtype=np.float64)

    x_wp = wp.array(x_np, dtype=wp.float64, device=DEVICE)
    y_wp = wp.array(y_np, dtype=wp.float64, device=DEVICE)

    # Allocate conserved & primitive arrays
    def alloc2d():
        return wp.zeros((NXG, NYG), dtype=wp.float64, device=DEVICE)

    ar1 = alloc2d(); ar2 = alloc2d()
    ru  = alloc2d(); rv  = alloc2d()
    E   = alloc2d(); al  = alloc2d()
    u_p = alloc2d(); v_p = alloc2d(); p_p = alloc2d()
    al_prim = alloc2d()   # ADD this new array
    ar1_old = alloc2d(); ar2_old = alloc2d()
    ru_old  = alloc2d(); rv_old  = alloc2d()
    E_old   = alloc2d(); al_old  = alloc2d()

    # Flux & source arrays
    fx0=alloc2d(); fx1=alloc2d(); fx2=alloc2d()
    fx3=alloc2d(); fx4=alloc2d(); fx5=alloc2d(); srcx=alloc2d()
    gy0=alloc2d(); gy1=alloc2d(); gy2=alloc2d()
    gy3=alloc2d(); gy4=alloc2d(); gy5=alloc2d(); srcy=alloc2d()
    fxl0=alloc2d(); fxl1=alloc2d(); fxl2=alloc2d()
    fxl3=alloc2d(); fxl4=alloc2d(); fxl5=alloc2d(); srcxl=alloc2d()
    gyl0=alloc2d(); gyl1=alloc2d(); gyl2=alloc2d()
    gyl3=alloc2d(); gyl4=alloc2d(); gyl5=alloc2d(); srcyl=alloc2d()
    theta_cell = alloc2d()

    dt_buf = wp.zeros(1, dtype=wp.float64, device=DEVICE)

    restart_loaded = args.restart != ""
    restart_time = 0.0
    restart_step = 0

    if restart_loaded:
        (ar1_np, ar2_np, ru_np, rv_np, E_np, al_np,
         u_np, v_np, p_np, restart_time_file) = load_restart_host(args.restart, x_np, y_np)
        restart_time = restart_time_file if args.restart_time is None else float(args.restart_time)
        restart_step = infer_restart_step(args.restart) if args.restart_step is None else int(args.restart_step)
        wp.copy(ar1, wp.array(ar1_np, dtype=wp.float64, device=DEVICE))
        wp.copy(ar2, wp.array(ar2_np, dtype=wp.float64, device=DEVICE))
        wp.copy(ru,  wp.array(ru_np,  dtype=wp.float64, device=DEVICE))
        wp.copy(rv,  wp.array(rv_np,  dtype=wp.float64, device=DEVICE))
        wp.copy(E,   wp.array(E_np,   dtype=wp.float64, device=DEVICE))
        wp.copy(al,  wp.array(al_np,  dtype=wp.float64, device=DEVICE))
        wp.copy(u_p, wp.array(u_np,   dtype=wp.float64, device=DEVICE))
        wp.copy(v_p, wp.array(v_np,   dtype=wp.float64, device=DEVICE))
        wp.copy(p_p, wp.array(p_np,   dtype=wp.float64, device=DEVICE))
        print(f"Restart: {args.restart}  t={restart_time:.6e}  step={restart_step}")
    else:
        wp.launch(ic_kernel2,
                  dim=(NXG, NYG),
                  inputs=[ar1, ar2, ru, rv, E, al, u_p, v_p, p_p, x_wp, y_wp],
                  device=DEVICE)

    # Helper lambdas
    def copy_old():
        wp.copy(ar1_old, ar1); wp.copy(ar2_old, ar2)
        wp.copy(ru_old,  ru);  wp.copy(rv_old,  rv)
        wp.copy(E_old,   E);   wp.copy(al_old,  al)

    def apply_bc():
        wp.launch(bc_x_kernel, dim=(G, NYG),
                  inputs=[ar1,ar2,ru,rv,E,al,u_p,v_p,p_p, G, NX, NY],
                  device=DEVICE)
        wp.launch(bc_y_kernel, dim=(NXG, G),
                  inputs=[ar1,ar2,ru,rv,E,al,u_p,v_p,p_p, G, NY],
                  device=DEVICE)

    def enforce_admissible():
        if USE_EMERGENCY_REPAIR:
            wp.launch(enforce_admissible_kernel, dim=(NXG, NYG),
                      inputs=[ar1, ar2, ru, rv, E, al],
                      device=DEVICE)

    def update_prims():
        wp.launch(cons_to_prim_kernel, dim=(NXG, NYG),
              inputs=[ar1,ar2,ru,rv,E,al, u_p,v_p,p_p],
              device=DEVICE)
        wp.copy(al_prim, al)   # ADD: snapshot clamped al after prim conversion

    def compute_fluxes():
        # x-fluxes: interfaces G-1 .. G+NX-1  (NX+1 total), for all NY interior cells
        wp.launch(fx_kernel, dim=(NX+1, NY),
                  inputs=[ar1,ar2,ru,rv,p_p,al,
                          fx0,fx1,fx2,fx3,fx4,fx5,srcx,
                          G-1, G],    # ofs_i=G-1 (first interface), ofs_j=G
                  device=DEVICE)
        # y-fluxes
        wp.launch(gy_kernel, dim=(NX, NY+1),
                  inputs=[ar1,ar2,ru,rv,p_p,al,
                          gy0,gy1,gy2,gy3,gy4,gy5,srcy,
                          G, G-1],    # ofs_i=G, ofs_j=G-1
                  device=DEVICE)

    def compute_low_fluxes():
        wp.launch(fx_low_kernel, dim=(NX+1, NY),
                  inputs=[ar1,ar2,ru,rv,p_p,al,
                          fxl0,fxl1,fxl2,fxl3,fxl4,fxl5,srcxl,
                          G-1, G],
                  device=DEVICE)
        wp.launch(gy_low_kernel, dim=(NX, NY+1),
                  inputs=[ar1,ar2,ru,rv,p_p,al,
                          gyl0,gyl1,gyl2,gyl3,gyl4,gyl5,srcyl,
                          G, G-1],
                  device=DEVICE)

    def limit_fluxes(dt_v):

        for stage in range(3):
            wp.launch(pp_cell_theta_kernel, dim=(NX, NY),
                      inputs=[ar1,ar2,ru,rv,E,al,al_prim,
                              fx0,fx1,fx2,fx3,fx4,fx5,srcx,
                              gy0,gy1,gy2,gy3,gy4,gy5,srcy,
                              fxl0,fxl1,fxl2,fxl3,fxl4,fxl5,srcxl,
                              gyl0,gyl1,gyl2,gyl3,gyl4,gyl5,srcyl,
                              theta_cell,
                              wp.float64(DX_V), wp.float64(DY_V), wp.float64(dt_v),
                              G, G, stage],
                      device=DEVICE)
            wp.launch(pp_blend_x_flux_kernel, dim=(NX+1, NY),
                      inputs=[fx0,fx1,fx2,fx3,fx4,fx5,srcx,
                              fxl0,fxl1,fxl2,fxl3,fxl4,fxl5,srcxl,
                              theta_cell, G-1, G, G, NX],
                      device=DEVICE)
            wp.launch(pp_blend_y_flux_kernel, dim=(NX, NY+1),
                      inputs=[gy0,gy1,gy2,gy3,gy4,gy5,srcy,
                              gyl0,gyl1,gyl2,gyl3,gyl4,gyl5,srcyl,
                              theta_cell, G, G-1, G, NY],
                      device=DEVICE)

    def rk3_update(alpha_rk, beta_rk, dt_v):
        wp.launch(rk3_kernel, dim=(NX, NY),
                  inputs=[ar1_old,ar2_old,ru_old,rv_old,E_old,al_old,
                          ar1,ar2,ru,rv,E,al, al_prim,  # al used as primitive α₁ for source
                          fx0,fx1,fx2,fx3,fx4,fx5,srcx,
                          gy0,gy1,gy2,gy3,gy4,gy5,srcy,
                          wp.float64(DX_V), wp.float64(DY_V),
                          wp.float64(dt_v),
                          wp.float64(alpha_rk), wp.float64(beta_rk),
                          G, G],
                  device=DEVICE)

    def get_dt():
        dt_np = np.array([1.0e30], dtype=np.float64)
        wp.copy(dt_buf, wp.array(dt_np, dtype=wp.float64, device=DEVICE))
        wp.launch(timestep_kernel, dim=(NX, NY),
                  inputs=[ar1,ar2,u_p,v_p,p_p,al,
                          wp.float64(DX_V), wp.float64(DY_V),
                          dt_buf, G, G],
                  device=DEVICE)
        wp.synchronize()
        return CFL_NUM * dt_buf.numpy()[0]

    # ── Initial output ───────────────────────────────────────────────────────
    enforce_admissible()
    apply_bc()
    update_prims()
    apply_bc()

    if not restart_loaded:
        save_npz(ar1.numpy(), ar2.numpy(), u_p.numpy(), v_p.numpy(),
                 p_p.numpy(), al.numpy(), x_np, y_np, 0, 0.0)
        save_png(ar1.numpy(), ar2.numpy(), u_p.numpy(), v_p.numpy(),
                 p_p.numpy(), al.numpy(), x_np, y_np, 0, 0.0)

    # ── Time loop ────────────────────────────────────────────────────────────
    time_sim = restart_time
    step     = restart_step
    save_idx = 0
    save_flags = [time_sim >= ts - 1.0e-14 for ts in T_SAVES]

    dt = get_dt()
    t0_wall = tm.time()

    while time_sim < T_END_V - 1.0e-14:

        # Check if we need to hit a save time exactly
        for k, ts in enumerate(T_SAVES):
            if not save_flags[k] and time_sim + dt > ts:
                dt = ts - time_sim

        if time_sim + dt > T_END_V:
            dt = T_END_V - time_sim

        copy_old()

        # Stage 1: Q(1) = Q(n) + dt*R(Q(n))
        apply_bc()
        compute_fluxes()
        compute_low_fluxes()
        limit_fluxes(dt)
        rk3_update(0.0, 1.0, dt)
        enforce_admissible()
        update_prims()

        # Stage 2: Q(2) = 3/4*Q(n) + 1/4*(Q(1) + dt*R(Q(1)))
        apply_bc()
        compute_fluxes()
        compute_low_fluxes()
        limit_fluxes(dt)
        rk3_update(0.75, 0.25, dt)
        enforce_admissible()
        update_prims()

        # Stage 3: Q(n+1) = 1/3*Q(n) + 2/3*(Q(2) + dt*R(Q(2)))
        apply_bc()
        compute_fluxes()
        compute_low_fluxes()
        limit_fluxes(dt)
        rk3_update(1.0/3.0, 2.0/3.0, dt)
        enforce_admissible()
        update_prims()

        time_sim += dt
        step     += 1
        dt        = get_dt()

        if step % 200 == 0:
            elapsed = tm.time() - t0_wall
            print(f"  step {step:5d}  t={time_sim:.4e}  dt={dt:.3e}  wall={elapsed:.1f}s")

        # Save outputs at target times
        for k, ts in enumerate(T_SAVES):
            if not save_flags[k] and abs(time_sim - ts) < 0.5*dt:
                ar1_np = ar1.numpy(); ar2_np = ar2.numpy()
                save_npz(ar1_np, ar2_np, u_p.numpy(), v_p.numpy(),
                         p_p.numpy(), al.numpy(), x_np, y_np, step, time_sim)
                save_png(ar1_np, ar2_np, u_p.numpy(), v_p.numpy(),
                         p_p.numpy(), al.numpy(), x_np, y_np, step, time_sim)
                save_flags[k] = True

    # Final output
    ar1_np = ar1.numpy(); ar2_np = ar2.numpy()
    save_npz(ar1_np, ar2_np, u_p.numpy(), v_p.numpy(),
             p_p.numpy(), al.numpy(), x_np, y_np, step, time_sim)
    save_png(ar1_np, ar2_np, u_p.numpy(), v_p.numpy(),
             p_p.numpy(), al.numpy(), x_np, y_np, step, time_sim)

    print(f"Done. Total steps={step}, t={time_sim:.4e}  wall={tm.time()-t0_wall:.1f}s")


if __name__ == "__main__":
    main()
