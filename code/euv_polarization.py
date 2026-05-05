"""
Mo/Si EUV Aperiodic — Polarization-Aware Optimization
Two parallel runs:
  (A) Unpolarized objective: J = mean(R_avg) + λ·min(R_avg),  R_avg = (R_TE+R_TM)/2
  (B) Polarized   objective: J = mean(R_TE)  + λ·min(R_TE)
For each design we evaluate R_TE, R_TM, R_avg → reproduce the paper's split panels.

Multistart K=3, λ ∈ {0, 1, 2}; smaller window list to keep wall time ≤ ~30 min.
Best design per (run, window) = max R_min in the chosen polarization (matches the
paper's "flat plateau" goal).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings, time, pickle
warnings.filterwarnings("ignore")

LAMBDA_NM   = 13.5
THETA0_DEG  = 6.0
N_MO        = 0.921 + 1j * 0.0064
N_SI        = 0.999 + 1j * 0.0018
N_VAC       = 1.0 + 0j
N_BILAYERS  = 40

WINDOWS  = [1.0, 3.0, 5.0, 7.0, 10.0, 13.0, 15.0]
LAMBDAS  = [0.0, 1.0, 2.0]
K_STARTS = 3


# ── TMM ───────────────────────────────────────────────────────────────
def build_stack(bilayers, top_mo=True):
    d = [0.0]; n = [N_VAC]
    for d_pair, gamma in bilayers:
        if top_mo:
            d += [d_pair*gamma, d_pair*(1-gamma)]; n += [N_MO, N_SI]
        else:
            d += [d_pair*(1-gamma), d_pair*gamma]; n += [N_SI, N_MO]
    d.append(0.0); n.append(N_SI)
    return np.array(d), np.array(n)


def _rec_r(n_list, kz, d_list, pol):
    N = len(n_list)
    def rij(i):
        if pol == 's':
            return (kz[i]-kz[i+1]) / (kz[i]+kz[i+1])
        return (n_list[i+1]**2*kz[i] - n_list[i]**2*kz[i+1]) / \
               (n_list[i+1]**2*kz[i] + n_list[i]**2*kz[i+1])
    r = rij(N-2)
    for i in range(N-3, -1, -1):
        ri = rij(i)
        phi = kz[i+1] * d_list[i+1]
        r = (ri + r*np.exp(2j*phi)) / (1 + ri*r*np.exp(2j*phi))
    return r


def R_te_tm(bilayers, theta_deg):
    """Return (R_TE, R_TM) for given bilayer stack and angle."""
    d_list, n_list = build_stack(bilayers)
    k0 = 2*np.pi / LAMBDA_NM
    sin0 = np.sin(np.deg2rad(theta_deg))
    kz = k0 * np.sqrt(n_list**2 - sin0**2 + 0j)
    Rs = abs(_rec_r(n_list, kz, d_list, 's'))**2     # TE
    Rp = abs(_rec_r(n_list, kz, d_list, 'p'))**2     # TM
    return float(Rs), float(Rp)


def R_obj(bilayers, theta_deg, pol_mode):
    """pol_mode: 'te' | 'tm' | 'avg'"""
    Rs, Rp = R_te_tm(bilayers, theta_deg)
    if pol_mode == 'te':  return Rs
    if pol_mode == 'tm':  return Rp
    return 0.5*(Rs + Rp)


def sweep_theta(bl, th_arr, pol_mode='avg'):
    return np.array([R_obj(bl, t, pol_mode) for t in th_arr])


def J_obj(bl, w, lam, pol_mode, npts=11):
    th = np.linspace(THETA0_DEG-w, THETA0_DEG+w, npts)
    R_arr = sweep_theta(bl, th, pol_mode)
    return float(np.mean(R_arr)) + lam*float(np.min(R_arr))


# ── Optimization ──────────────────────────────────────────────────────
def opt_periodic(pol_mode='te'):
    """Optimize periodic for max R at θ₀ in given polarization."""
    def neg(p):
        d, g = p
        bl = [(np.clip(d,4,14), np.clip(g,0.15,0.70))]*N_BILAYERS
        return -R_obj(bl, THETA0_DEG, pol_mode)
    res = minimize(neg, [6.94, 0.39], method='Nelder-Mead',
                   options={'xatol':1e-5,'fatol':1e-8,'maxiter':800})
    d, g = np.clip(res.x[0],4,14), np.clip(res.x[1],0.15,0.70)
    return [(d, g)]*N_BILAYERS, d, g


def greedy_grow(w, lam, d0, g0, pol_mode):
    bilayers = []
    for _ in range(N_BILAYERS):
        def neg(p, _bl=bilayers):
            d, g = p
            full = _bl + [(np.clip(d,4,14), np.clip(g,0.15,0.70))]
            return -J_obj(full, w, lam, pol_mode, npts=11)
        res = minimize(neg, [d0, g0], method='Nelder-Mead',
                       options={'xatol':1e-4,'fatol':1e-7,'maxiter':180,'adaptive':True})
        bilayers.append((np.clip(res.x[0],4,14), np.clip(res.x[1],0.15,0.70)))
    return bilayers


def coord_refine(bl, w, lam, pol_mode, passes=1):
    bl = list(bl)
    for _ in range(passes):
        for i in range(len(bl)):
            pre, suf = bl[:i], bl[i+1:]
            def neg(p, _pre=pre, _suf=suf):
                d, g = p
                full = _pre + [(np.clip(d,4,14), np.clip(g,0.15,0.70))] + _suf
                return -J_obj(full, w, lam, pol_mode, npts=11)
            res = minimize(neg, list(bl[i]), method='Nelder-Mead',
                           options={'xatol':1e-4,'fatol':1e-7,'maxiter':120,'adaptive':True})
            bl[i] = (np.clip(res.x[0],4,14), np.clip(res.x[1],0.15,0.70))
    return bl


def evaluate_all(bl, w):
    """Evaluate TE, TM, AVG reflectivity over the window and at θ₀."""
    th = np.linspace(THETA0_DEG-w, THETA0_DEG+w, 161)
    R_te = sweep_theta(bl, th, 'te')
    R_tm = sweep_theta(bl, th, 'tm')
    R_av = 0.5*(R_te + R_tm)
    return {'bl': bl, 'thetas': th,
            'R_te': R_te, 'R_tm': R_tm, 'R_avg': R_av,
            'mean_te': float(R_te.mean()),  'min_te':  float(R_te.min()),
            'mean_tm': float(R_tm.mean()),  'min_tm':  float(R_tm.min()),
            'mean_av': float(R_av.mean()),  'min_av':  float(R_av.min())}


# ── Main ─────────────────────────────────────────────────────────────
def run_grid(pol_mode, d0, g0, label):
    print(f"\n═══ Run: {label}  (objective uses {pol_mode}) ═══")
    cands = {}
    for w in WINDOWS:
        for lam in LAMBDAS:
            for k in range(K_STARTS):
                sc = np.linspace(0.94, 1.06, K_STARTS)[k]
                d0k = np.clip(d0*sc, 4, 14)
                g0k = np.clip(g0*(1.0 + 0.02*(k - (K_STARTS-1)/2)), 0.15, 0.70)
                t0 = time.time()
                bl = greedy_grow(w, lam, d0k, g0k, pol_mode)
                bl = coord_refine(bl, w, lam, pol_mode, passes=1)
                e = evaluate_all(bl, w)
                e['lam'] = lam; e['k'] = k
                cands[(w, lam, k)] = e
                key_min = e['min_te'] if pol_mode=='te' else e['min_av']
                key_mean = e['mean_te'] if pol_mode=='te' else e['mean_av']
                print(f"  ±{w:>4}° λ={lam:.1f} k={k}  "
                      f"mean={key_mean*100:.2f}% min={key_min*100:.2f}%  "
                      f"({time.time()-t0:.1f}s)")
    return cands


def pick_best(cands, pol_mode):
    """Pick design with max R_min in the optimized polarization."""
    key = 'min_te' if pol_mode == 'te' else 'min_av'
    best = {}
    for w in WINDOWS:
        c = [cands[(w, l, k)] for l in LAMBDAS for k in range(K_STARTS)]
        best[w] = max(c, key=lambda e: e[key])
    return best


def main():
    t0_all = time.time()

    # Periodic baselines for each polarization (each is a "fair" reference)
    print("Optimizing periodic baselines...")
    bl_per_te,  d_te, g_te = opt_periodic('te')
    bl_per_avg, d_av, g_av = opt_periodic('avg')
    print(f"  Periodic (TE-opt):  d={d_te:.3f}  γ={g_te:.3f}  R_TE@θ₀={R_obj(bl_per_te,THETA0_DEG,'te')*100:.2f}%")
    print(f"  Periodic (avg-opt): d={d_av:.3f}  γ={g_av:.3f}  R_avg@θ₀={R_obj(bl_per_avg,THETA0_DEG,'avg')*100:.2f}%")

    # Use TE-optimized periodic seed for both grids (consistent starting point)
    d0, g0 = d_te, g_te

    cands_pol   = run_grid('te',  d0, g0, 'POLARIZED Aperiodic')
    cands_unpol = run_grid('avg', d0, g0, 'UNPOLARIZED Aperiodic')

    best_pol   = pick_best(cands_pol,   'te')
    best_unpol = pick_best(cands_unpol, 'avg')

    # Save
    with open('euv_pol_results.pkl', 'wb') as f:
        pickle.dump({'best_pol': best_pol, 'best_unpol': best_unpol,
                     'bl_per_te': bl_per_te, 'bl_per_avg': bl_per_avg,
                     'd_te': d_te, 'g_te': g_te,
                     'cands_pol': cands_pol, 'cands_unpol': cands_unpol}, f)

    # ── Summary table ──
    print("\n══════════════ SUMMARY (best per window, max R_min in optimized pol) ══════════════")
    print(f"{'window':>8}  | {'Polarized: mean_TE / min_TE / mean_TM / min_TM':<55} | "
          f"{'Unpolarized: mean_av / min_av / mean_TE / mean_TM':<55}")
    for w in WINDOWS:
        bp = best_pol[w]; bu = best_unpol[w]
        print(f"  ±{w:>4}°   "
              f"  TE: {bp['mean_te']*100:.1f}/{bp['min_te']*100:.1f}  "
              f"TM: {bp['mean_tm']*100:.1f}/{bp['min_tm']*100:.1f}    |    "
              f"AV: {bu['mean_av']*100:.1f}/{bu['min_av']*100:.1f}  "
              f"TE: {bu['mean_te']*100:.1f}  TM: {bu['mean_tm']*100:.1f}")

    # ── PLOT 1: Three-panel reproduction (Periodic | Unpolarized | Polarized) ──
    print("\nPlotting paper-style triple panel...")
    w_show = 13.0
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    th_full = np.linspace(THETA0_DEG-15, THETA0_DEG+15, 401)

    # Panel 1: Conventional Periodic (use TE-opt periodic, plot TE+TM)
    ax = axes[0]
    R_te_per = sweep_theta(bl_per_te, th_full, 'te')
    R_tm_per = sweep_theta(bl_per_te, th_full, 'tm')
    ax.plot(th_full, R_te_per, '-', lw=2, color='#1f77b4', label='TE Polarized')
    ax.plot(th_full, R_tm_per, '-', lw=2, color='#ff7f0e', label='TM Polarized')
    ax.set_title('Conventional Periodic', fontsize=14)
    ax.set_xlabel('Incident Angle (°)', fontsize=12)
    ax.set_ylabel('Reflectance', fontsize=12)
    ax.legend(fontsize=11); ax.grid(alpha=0.3); ax.set_ylim(0, 0.85)

    # Panel 2: Unpolarized Aperiodic (best at w_show, plot TE+TM)
    ax = axes[1]
    bu = best_unpol[w_show]
    th_eval = np.linspace(THETA0_DEG-15, THETA0_DEG+15, 401)
    R_te_u  = sweep_theta(bu['bl'], th_eval, 'te')
    R_tm_u  = sweep_theta(bu['bl'], th_eval, 'tm')
    ax.plot(th_eval, R_te_u, '-', lw=2, color='#1f77b4', label='TE Polarized')
    ax.plot(th_eval, R_tm_u, '-', lw=2, color='#ff7f0e', label='TM Polarized')
    ax.set_title(f'Unpolarized Aperiodic\n(opt: avg, ±{w_show}°)', fontsize=14)
    ax.set_xlabel('Incident Angle (°)', fontsize=12)
    ax.legend(fontsize=11); ax.grid(alpha=0.3); ax.set_ylim(0, 0.85)

    # Panel 3: Polarized Aperiodic (best at w_show, plot TE+TM)
    ax = axes[2]
    bp = best_pol[w_show]
    R_te_p  = sweep_theta(bp['bl'], th_eval, 'te')
    R_tm_p  = sweep_theta(bp['bl'], th_eval, 'tm')
    ax.plot(th_eval, R_te_p, '-', lw=2, color='#1f77b4', label='TE Polarized')
    ax.plot(th_eval, R_tm_p, '-', lw=2, color='#ff7f0e', label='TM Polarized')
    ax.set_title(f'Polarized Aperiodic\n(opt: TE, ±{w_show}°)', fontsize=14)
    ax.set_xlabel('Incident Angle (°)', fontsize=12)
    ax.legend(fontsize=11); ax.grid(alpha=0.3); ax.set_ylim(0, 0.85)

    plt.suptitle(f'Polarized vs Unpolarized Aperiodic Optimization (window ±{w_show}°)\n'
                 f'Mo/Si N=40, λ=13.5 nm, θ₀=6°', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_pol_triple.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig_pol_triple.png")

    # ── PLOT 2: window sweep — TE response of polarized aperiodic ──
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    cmap = plt.cm.plasma

    ax = axes[0]
    for i, w in enumerate(WINDOWS):
        bp = best_pol[w]
        col = cmap(i/(len(WINDOWS)-1))
        ax.plot(bp['thetas'], bp['R_te']*100, '-', lw=2, color=col,
                label=f'±{w}°')
    R_te_per_full = sweep_theta(bl_per_te, th_full, 'te')*100
    ax.plot(th_full, R_te_per_full, 'k:', lw=2, alpha=0.85, label='Periodic (TE)')
    ax.axvline(THETA0_DEG, color='gray', ls=':', lw=1)
    ax.set_xlabel('θ (°)', fontsize=12)
    ax.set_ylabel('R_TE (%)', fontsize=12)
    ax.set_title('Polarized Aperiodic: R_TE per window', fontsize=12)
    ax.legend(fontsize=9, ncol=2, loc='lower center'); ax.grid(alpha=0.3)
    ax.set_ylim(0, None)

    ax = axes[1]
    for i, w in enumerate(WINDOWS):
        bu = best_unpol[w]
        col = cmap(i/(len(WINDOWS)-1))
        ax.plot(bu['thetas'], bu['R_avg']*100, '-', lw=2, color=col,
                label=f'±{w}°')
    R_avg_per_full = sweep_theta(bl_per_te, th_full, 'avg')*100
    ax.plot(th_full, R_avg_per_full, 'k:', lw=2, alpha=0.85, label='Periodic (avg)')
    ax.axvline(THETA0_DEG, color='gray', ls=':', lw=1)
    ax.set_xlabel('θ (°)', fontsize=12)
    ax.set_ylabel('R_avg (%)', fontsize=12)
    ax.set_title('Unpolarized Aperiodic: R_avg per window', fontsize=12)
    ax.legend(fontsize=9, ncol=2, loc='lower center'); ax.grid(alpha=0.3)
    ax.set_ylim(0, None)

    plt.suptitle(f'Window Sweep — Best designs (max R_min in optimized polarization)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_pol_window_sweep.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig_pol_window_sweep.png")

    # ── PLOT 3: mean / min vs window in optimized polarization ──
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))

    ax = axes[0]
    pol_mean   = [best_pol[w]['mean_te']*100   for w in WINDOWS]
    pol_min    = [best_pol[w]['min_te']*100    for w in WINDOWS]
    unpol_mean = [best_unpol[w]['mean_av']*100 for w in WINDOWS]
    unpol_min  = [best_unpol[w]['min_av']*100  for w in WINDOWS]

    ax.plot(WINDOWS, pol_mean, 's-', color='#1f77b4', lw=2.5, ms=9, label='Polarized: R_TE mean')
    ax.plot(WINDOWS, pol_min,  '^-', color='#1f77b4', lw=2,   ms=8, alpha=0.7, label='Polarized: R_TE min')
    ax.plot(WINDOWS, unpol_mean, 's--', color='#ff7f0e', lw=2.5, ms=9, label='Unpolarized: R_avg mean')
    ax.plot(WINDOWS, unpol_min,  '^--', color='#ff7f0e', lw=2,   ms=8, alpha=0.7, label='Unpolarized: R_avg min')
    ax.set_xlabel('Window ±Δθ (°)', fontsize=12)
    ax.set_ylabel('Reflectivity (%)', fontsize=12)
    ax.set_title('Polarized vs Unpolarized: mean & min vs window', fontsize=12)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    # cross-evaluation: how does the polarized design fare under unpol illumination?
    ax = axes[1]
    pol_in_avg_mean = [best_pol[w]['mean_av']*100 for w in WINDOWS]
    pol_in_avg_min  = [best_pol[w]['min_av']*100  for w in WINDOWS]
    unpol_in_te_mean = [best_unpol[w]['mean_te']*100 for w in WINDOWS]
    unpol_in_te_min  = [best_unpol[w]['min_te']*100  for w in WINDOWS]

    ax.plot(WINDOWS, pol_in_avg_mean, 's-', color='#1f77b4', lw=2.5, ms=9,
            label='Polarized design: R_avg mean')
    ax.plot(WINDOWS, pol_in_avg_min,  '^-', color='#1f77b4', lw=2,   ms=8, alpha=0.7,
            label='Polarized design: R_avg min')
    ax.plot(WINDOWS, unpol_in_te_mean, 's--', color='#ff7f0e', lw=2.5, ms=9,
            label='Unpolarized design: R_TE mean')
    ax.plot(WINDOWS, unpol_in_te_min,  '^--', color='#ff7f0e', lw=2,   ms=8, alpha=0.7,
            label='Unpolarized design: R_TE min')
    ax.set_xlabel('Window ±Δθ (°)', fontsize=12)
    ax.set_ylabel('Reflectivity (%)', fontsize=12)
    ax.set_title('Cross-evaluation:\npolarized design under unpol light, vice versa', fontsize=12)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig_pol_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig_pol_metrics.png")

    print(f"\nTotal wall time: {time.time()-t0_all:.1f}s")


if __name__ == '__main__':
    main()
