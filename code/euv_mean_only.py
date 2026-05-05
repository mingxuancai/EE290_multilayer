"""
Mo/Si EUV Aperiodic Multilayer — Mean-Only Optimization (α=0)
Sweep 角度窗口宽度，纯优化 mean reflection。
Greedy + 2 passes coord refine 提升稳健性。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings, time
warnings.filterwarnings("ignore")

LAMBDA_NM = 13.5
THETA0_DEG = 6.0
N_MO  = 0.921 + 1j * 0.0064
N_SI  = 0.999 + 1j * 0.0018
N_VAC = 1.0 + 0j
N_BILAYERS = 40

ALPHA = 0.0
WINDOWS = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])


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


def R_stack(bilayers, theta_deg, top_mo=True):
    d_list, n_list = build_stack(bilayers, top_mo)
    k0 = 2*np.pi / LAMBDA_NM
    sin0 = np.sin(np.deg2rad(theta_deg))
    kz = k0 * np.sqrt(n_list**2 - sin0**2 + 0j)
    Rs = abs(_rec_r(n_list, kz, d_list, 's'))**2
    Rp = abs(_rec_r(n_list, kz, d_list, 'p'))**2
    return 0.5*(float(Rs)+float(Rp))


def sweep_theta(bilayers, theta_arr):
    return np.array([R_stack(bilayers, t) for t in theta_arr])


def J_mean(bilayers, window_deg, npts=11):
    thetas = np.linspace(THETA0_DEG-window_deg, THETA0_DEG+window_deg, npts)
    return float(np.mean(sweep_theta(bilayers, thetas)))


def opt_periodic():
    def neg(p):
        d, g = p
        bl = [(np.clip(d,4,14), np.clip(g,0.15,0.70))]*N_BILAYERS
        return -R_stack(bl, THETA0_DEG)
    res = minimize(neg, [6.94, 0.39], method='Nelder-Mead',
                   options={'xatol':1e-5,'fatol':1e-8,'maxiter':800})
    d, g = np.clip(res.x[0],4,14), np.clip(res.x[1],0.15,0.70)
    return [(d, g)]*N_BILAYERS, d, g


def greedy_grow(window_deg, d0, g0):
    bilayers = []
    for _ in range(N_BILAYERS):
        def neg(p, _bl=bilayers):
            d, g = p
            full = _bl + [(np.clip(d,4,14), np.clip(g,0.15,0.70))]
            return -J_mean(full, window_deg, npts=11)
        res = minimize(neg, [d0, g0], method='Nelder-Mead',
                       options={'xatol':1e-4,'fatol':1e-7,'maxiter':200,'adaptive':True})
        bilayers.append((np.clip(res.x[0],4,14), np.clip(res.x[1],0.15,0.70)))
    return bilayers


def coord_refine(bilayers, window_deg, passes=2):
    bl = list(bilayers)
    for p_i in range(passes):
        for i in range(len(bl)):
            pre, suf = bl[:i], bl[i+1:]
            def neg(p, _pre=pre, _suf=suf):
                d, g = p
                full = _pre + [(np.clip(d,4,14), np.clip(g,0.15,0.70))] + _suf
                return -J_mean(full, window_deg, npts=11)
            res = minimize(neg, list(bl[i]), method='Nelder-Mead',
                           options={'xatol':1e-4,'fatol':1e-7,'maxiter':150,'adaptive':True})
            bl[i] = (np.clip(res.x[0],4,14), np.clip(res.x[1],0.15,0.70))
    return bl


def evaluate(bl, window_deg):
    thetas = np.linspace(THETA0_DEG-window_deg, THETA0_DEG+window_deg, 81)
    R_arr = sweep_theta(bl, thetas)
    return {'bl': bl,
            'R_peak': R_stack(bl, THETA0_DEG),
            'R_mean': float(np.mean(R_arr)),
            'R_min':  float(np.min(R_arr)),
            'thetas': thetas, 'R_arr': R_arr}


def main():
    t_total = time.time()
    print("Optimizing periodic baseline...")
    bl_per, d0, g0 = opt_periodic()
    print(f"  d={d0:.3f} nm, γ={g0:.3f}, R_peak={R_stack(bl_per,THETA0_DEG)*100:.2f}%\n")

    # FWHM
    thetas_full = np.linspace(0, 25, 500)
    R_full = sweep_theta(bl_per, thetas_full)
    above = thetas_full[R_full >= R_full.max()/2]
    fwhm = above[-1] - above[0]
    print(f"Periodic FWHM = {fwhm:.2f}°, half-width = {fwhm/2:.2f}°\n")

    print(f"═══ Window sweep @ α={ALPHA} (pure mean maximization) ═══")
    results = {}
    for w in WINDOWS:
        t0 = time.time()
        bl = greedy_grow(w, d0, g0)
        bl = coord_refine(bl, w, passes=2)
        e_ap  = evaluate(bl, w)
        e_per = evaluate(bl_per, w)
        results[w] = {'ap': e_ap, 'per': e_per}
        print(f"  ±{w:>4}°  "
              f"mean_per={e_per['R_mean']*100:.2f}%  "
              f"mean_ap={e_ap['R_mean']*100:.2f}%  "
              f"Δmean={(e_ap['R_mean']-e_per['R_mean'])*100:+.2f}%  "
              f"peak_ap={e_ap['R_peak']*100:.2f}%  "
              f"min_ap={e_ap['R_min']*100:.2f}%  "
              f"({time.time()-t0:.1f}s)")

    # ── PLOT ──────────────────────────────────────────────────────────────
    print("\nPlotting...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))

    per_mean = np.array([results[w]['per']['R_mean'] for w in WINDOWS])*100
    per_peak = np.array([results[w]['per']['R_peak'] for w in WINDOWS])*100
    ap_mean  = np.array([results[w]['ap']['R_mean']  for w in WINDOWS])*100
    ap_peak  = np.array([results[w]['ap']['R_peak']  for w in WINDOWS])*100
    ap_min   = np.array([results[w]['ap']['R_min']   for w in WINDOWS])*100
    per_min  = np.array([results[w]['per']['R_min']  for w in WINDOWS])*100

    # ── Panel 1: mean vs window
    ax = axes[0]
    ax.plot(WINDOWS, per_mean, 'o:', color='#1a9641', lw=1.8, ms=8, label='Periodic mean')
    ax.plot(WINDOWS, ap_mean, 's-',  color='#1a9641', lw=2.5, ms=9, label='Aperiodic mean (α=0)')
    ax.plot(WINDOWS, ap_peak, '^--', color='#d73027', lw=1.6, ms=7, alpha=0.8, label='Aperiodic peak')
    ax.plot(WINDOWS, ap_min,  'v--', color='#756bb1', lw=1.4, ms=6, alpha=0.8, label='Aperiodic min')
    ax.axvline(fwhm/2, color='orange', ls='--', lw=1.4, label=f'Bragg half-width {fwhm/2:.1f}°')
    ax.set_xlabel('Window ±Δθ (°)', fontsize=11)
    ax.set_ylabel('Reflectivity (%)', fontsize=11)
    ax.set_title(f'Mean-Only Optimization (α=0)\nReflectivity vs Window', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Panel 2: Δmean gain
    ax = axes[1]
    dmean = ap_mean - per_mean
    colors = ['#1a9641' if d >= 0 else '#d73027' for d in dmean]
    ax.bar(WINDOWS, dmean, width=0.55, color=colors, alpha=0.75,
           edgecolor='k', linewidth=0.6)
    for w, g in zip(WINDOWS, dmean):
        ax.annotate(f'{g:+.2f}', (w, g), fontsize=8,
                    textcoords='offset points',
                    xytext=(0, 8 if g >= 0 else -14),
                    ha='center',
                    color='#1a9641' if g >= 0 else '#d73027')
    ax.axhline(0, color='k', lw=1.2)
    ax.axvline(fwhm/2, color='orange', ls='--', lw=1.4,
               label=f'Bragg half-width {fwhm/2:.1f}°')
    ax.fill_betweenx([-6, 10], 0, fwhm/2, alpha=0.07, color='blue', label='Plateau')
    ax.fill_betweenx([-6, 10], fwhm/2, 17, alpha=0.07, color='red', label='Slope')
    ax.set_xlabel('Window ±Δθ (°)', fontsize=11)
    ax.set_ylabel('ΔR_mean: Aperiodic − Periodic (%)', fontsize=11)
    ax.set_title(f'Gain from Aperiodic (α=0) vs Window', fontsize=10)
    ax.set_xlim(0, 16)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    # ── Panel 3: R(θ) overlay
    ax = axes[2]
    cmap = plt.cm.plasma
    for i, w in enumerate(WINDOWS):
        col = cmap(i/(len(WINDOWS)-1))
        e_ap = results[w]['ap']
        ax.plot(e_ap['thetas'], e_ap['R_arr']*100, '-', lw=1.6, color=col,
                label=f'±{w}°')
    # periodic as reference overlay
    th_ref = np.linspace(THETA0_DEG-15, THETA0_DEG+15, 200)
    R_ref  = sweep_theta(bl_per, th_ref)
    ax.plot(th_ref, R_ref*100, 'k:', lw=2, alpha=0.8, label='Periodic (ref)')
    ax.axvline(THETA0_DEG, color='gray', ls=':', lw=1)
    ax.set_xlabel('θ (°)', fontsize=11)
    ax.set_ylabel('R (%)', fontsize=11)
    ax.set_title('R(θ) of α=0 aperiodic solutions', fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc='lower center')
    ax.grid(alpha=0.3); ax.set_ylim(0, None)

    plt.suptitle(f'Mo/Si EUV Aperiodic — Pure Mean Optimization (α=0), Window Sweep  '
                 f'(N={N_BILAYERS}, λ=13.5 nm, θ₀={THETA0_DEG}°)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_mean_only.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig_mean_only.png")
    print(f"\nTotal wall time: {time.time()-t_total:.1f}s")


if __name__ == '__main__':
    main()
