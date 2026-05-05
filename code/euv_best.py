"""
Mo/Si EUV Aperiodic — Best-Design Window Sweep
对每个 window 运行 λ × K-multistart 的网格，挑两种"最优"设计：
  (A) max R_mean 作主图
  (B) max R_min  作副图
每图 4 个子图：peak / mean / min vs window，加 R(θ) overlay。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings, time, pickle
warnings.filterwarnings("ignore")

LAMBDA_NM = 13.5
THETA0_DEG = 6.0
N_MO  = 0.921 + 1j * 0.0064
N_SI  = 0.999 + 1j * 0.0018
N_VAC = 1.0 + 0j
N_BILAYERS = 40

WINDOWS  = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 13.0, 15.0]
LAMBDAS  = [0.0, 0.5, 1.0, 2.0]
K_STARTS = 4


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


def R_stack(bilayers, theta_deg):
    d_list, n_list = build_stack(bilayers)
    k0 = 2*np.pi / LAMBDA_NM
    sin0 = np.sin(np.deg2rad(theta_deg))
    kz = k0 * np.sqrt(n_list**2 - sin0**2 + 0j)
    Rs = abs(_rec_r(n_list, kz, d_list, 's'))**2
    Rp = abs(_rec_r(n_list, kz, d_list, 'p'))**2
    return 0.5*(float(Rs)+float(Rp))


def sweep_theta(bilayers, theta_arr):
    return np.array([R_stack(bilayers, t) for t in theta_arr])


def J_obj(bilayers, window_deg, lam, npts=11):
    thetas = np.linspace(THETA0_DEG-window_deg, THETA0_DEG+window_deg, npts)
    R_arr = sweep_theta(bilayers, thetas)
    return float(np.mean(R_arr)) + lam * float(np.min(R_arr))


def opt_periodic():
    def neg(p):
        d, g = p
        bl = [(np.clip(d,4,14), np.clip(g,0.15,0.70))]*N_BILAYERS
        return -R_stack(bl, THETA0_DEG)
    res = minimize(neg, [6.94, 0.39], method='Nelder-Mead',
                   options={'xatol':1e-5,'fatol':1e-8,'maxiter':800})
    d, g = np.clip(res.x[0],4,14), np.clip(res.x[1],0.15,0.70)
    return [(d, g)]*N_BILAYERS, d, g


def greedy_grow(window_deg, lam, d0, g0):
    bilayers = []
    for _ in range(N_BILAYERS):
        def neg(p, _bl=bilayers):
            d, g = p
            full = _bl + [(np.clip(d,4,14), np.clip(g,0.15,0.70))]
            return -J_obj(full, window_deg, lam, npts=11)
        res = minimize(neg, [d0, g0], method='Nelder-Mead',
                       options={'xatol':1e-4,'fatol':1e-7,'maxiter':180,'adaptive':True})
        bilayers.append((np.clip(res.x[0],4,14), np.clip(res.x[1],0.15,0.70)))
    return bilayers


def coord_refine(bilayers, window_deg, lam, passes=1):
    bl = list(bilayers)
    for _ in range(passes):
        for i in range(len(bl)):
            pre, suf = bl[:i], bl[i+1:]
            def neg(p, _pre=pre, _suf=suf):
                d, g = p
                full = _pre + [(np.clip(d,4,14), np.clip(g,0.15,0.70))] + _suf
                return -J_obj(full, window_deg, lam, npts=11)
            res = minimize(neg, list(bl[i]), method='Nelder-Mead',
                           options={'xatol':1e-4,'fatol':1e-7,'maxiter':120,'adaptive':True})
            bl[i] = (np.clip(res.x[0],4,14), np.clip(res.x[1],0.15,0.70))
    return bl


def evaluate(bl, window_deg):
    thetas = np.linspace(THETA0_DEG-window_deg, THETA0_DEG+window_deg, 161)
    R_arr = sweep_theta(bl, thetas)
    return {'bl': bl,
            'R_peak': R_stack(bl, THETA0_DEG),
            'R_mean': float(np.mean(R_arr)),
            'R_min':  float(np.min(R_arr)),
            'thetas': thetas, 'R_arr': R_arr}


def run_grid(d0, g0):
    """对所有 (window, λ, k) 跑一遍 greedy+refine。"""
    all_cands = {}   # (w, λ, k) -> eval dict (+ lam tag)
    for w in WINDOWS:
        for lam in LAMBDAS:
            for k in range(K_STARTS):
                sc = np.linspace(0.94, 1.06, K_STARTS)[k]
                d0k = np.clip(d0*sc, 4, 14)
                g0k = np.clip(g0*(1.0 + 0.02*(k - (K_STARTS-1)/2)), 0.15, 0.70)
                t0 = time.time()
                bl = greedy_grow(w, lam, d0k, g0k)
                bl = coord_refine(bl, w, lam, passes=1)
                e = evaluate(bl, w)
                e['lam'] = lam; e['k'] = k; e['t'] = time.time()-t0
                all_cands[(w, lam, k)] = e
                print(f"  ±{w:>4}° λ={lam:.1f} k={k}  "
                      f"mean={e['R_mean']*100:.2f}% min={e['R_min']*100:.2f}% "
                      f"peak={e['R_peak']*100:.2f}%  ({e['t']:.1f}s)")
    return all_cands


def pick_best(all_cands, metric):
    """对每个 window，按 metric 选最佳。metric='R_mean' 或 'R_min'。"""
    best = {}
    for w in WINDOWS:
        cands = [all_cands[(w, lam, k)] for lam in LAMBDAS for k in range(K_STARTS)]
        best[w] = max(cands, key=lambda e: e[metric])
    return best


# ── PLOT ───────────────────────────────────────────────────────────────────
def plot_best(best, per_evals, criterion_name, outpath, fwhm):
    """
    best: dict w -> evaluate(bl_best, w) (+lam, +k tags)
    per_evals: dict w -> evaluate(bl_per, w)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 数据
    per_peak = np.array([per_evals[w]['R_peak']*100 for w in WINDOWS])
    per_mean = np.array([per_evals[w]['R_mean']*100 for w in WINDOWS])
    per_min  = np.array([per_evals[w]['R_min']*100  for w in WINDOWS])
    ap_peak  = np.array([best[w]['R_peak']*100 for w in WINDOWS])
    ap_mean  = np.array([best[w]['R_mean']*100 for w in WINDOWS])
    ap_min   = np.array([best[w]['R_min']*100  for w in WINDOWS])
    chosen_lam = [best[w]['lam'] for w in WINDOWS]

    # ── (0,0): peak
    ax = axes[0][0]
    ax.plot(WINDOWS, per_peak, 'o:', color='#d73027', lw=1.6, ms=7, label='Periodic peak')
    ax.plot(WINDOWS, ap_peak,  'o-', color='#d73027', lw=2.4, ms=9, label=f'Aperiodic peak ({criterion_name})')
    ax.axvline(fwhm/2, color='orange', ls='--', lw=1.2, label=f'Bragg half-width {fwhm/2:.1f}°')
    ax.set_xlabel('Window ±Δθ (°)')
    ax.set_ylabel('R_peak (%)')
    ax.set_title('Peak Reflectivity @ θ₀=6° vs Window', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── (0,1): mean
    ax = axes[0][1]
    ax.plot(WINDOWS, per_mean, 's:', color='#1a9641', lw=1.6, ms=7, label='Periodic mean')
    ax.plot(WINDOWS, ap_mean,  's-', color='#1a9641', lw=2.4, ms=9, label=f'Aperiodic mean ({criterion_name})')
    for w, m, l in zip(WINDOWS, ap_mean, chosen_lam):
        ax.annotate(f'λ*={l}', (w, m), fontsize=7,
                    textcoords='offset points', xytext=(0, 8), ha='center', color='#1a9641')
    ax.axvline(fwhm/2, color='orange', ls='--', lw=1.2, label=f'Bragg half-width {fwhm/2:.1f}°')
    ax.set_xlabel('Window ±Δθ (°)')
    ax.set_ylabel('R_mean (%)')
    ax.set_title('Mean Reflectivity over ±Window (λ* annotated)', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── (1,0): min
    ax = axes[1][0]
    ax.plot(WINDOWS, per_min, '^:', color='#756bb1', lw=1.6, ms=7, label='Periodic min')
    ax.plot(WINDOWS, ap_min,  '^-', color='#756bb1', lw=2.4, ms=9, label=f'Aperiodic min ({criterion_name})')
    ax.axvline(fwhm/2, color='orange', ls='--', lw=1.2, label=f'Bragg half-width {fwhm/2:.1f}°')
    ax.set_xlabel('Window ±Δθ (°)')
    ax.set_ylabel('R_min (%)')
    ax.set_title('Worst-case (Min) Reflectivity over ±Window', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── (1,1): R(θ) overlay
    ax = axes[1][1]
    cmap = plt.cm.plasma
    for i, w in enumerate(WINDOWS):
        e = best[w]
        col = cmap(i/(len(WINDOWS)-1))
        ax.plot(e['thetas'], e['R_arr']*100, '-', lw=1.6, color=col,
                label=f'±{w}° (λ*={e["lam"]})')
    # periodic reference
    per_bl = per_evals[WINDOWS[-1]]['bl']
    th_ref = np.linspace(THETA0_DEG-15, THETA0_DEG+15, 240)
    ax.plot(th_ref, sweep_theta(per_bl, th_ref)*100, 'k:', lw=2, alpha=0.8,
            label='Periodic (ref)')
    ax.axvline(THETA0_DEG, color='gray', ls=':', lw=1)
    ax.set_xlabel('θ (°)')
    ax.set_ylabel('R (%)')
    ax.set_title(f'R(θ) of best designs per window\n(criterion: {criterion_name})', fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc='lower center')
    ax.grid(alpha=0.3); ax.set_ylim(0, None)

    plt.suptitle(f'Mo/Si EUV Aperiodic — Best Design per Window  '
                 f'(criterion: {criterion_name})\n'
                 f'N={N_BILAYERS}, λ=13.5 nm, θ₀={THETA0_DEG}°, '
                 f'grid: λ∈{LAMBDAS}, K-multistart={K_STARTS}',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {outpath}")


def main():
    t_total = time.time()
    print("Optimizing periodic baseline...")
    bl_per, d0, g0 = opt_periodic()
    print(f"  d={d0:.3f} nm, γ={g0:.3f}, R_peak={R_stack(bl_per,THETA0_DEG)*100:.2f}%\n")

    thetas_full = np.linspace(0, 25, 500)
    R_full = sweep_theta(bl_per, thetas_full)
    above = thetas_full[R_full >= R_full.max()/2]
    fwhm = above[-1] - above[0]
    print(f"Periodic FWHM = {fwhm:.2f}°, half-width = {fwhm/2:.2f}°\n")

    # Periodic evaluations at each window (reference)
    per_evals = {w: evaluate(bl_per, w) for w in WINDOWS}

    n_total = len(WINDOWS) * len(LAMBDAS) * K_STARTS
    print(f"═══ Grid: {len(WINDOWS)} windows × {len(LAMBDAS)} λ × K={K_STARTS} "
          f"= {n_total} greedy runs ═══")
    all_cands = run_grid(d0, g0)

    # 保存原始结果
    with open('euv_best_results.pkl', 'wb') as f:
        pickle.dump({'all_cands': all_cands, 'per_evals': per_evals,
                     'fwhm': fwhm, 'd0': d0, 'g0': g0}, f)
    print("\nSaved raw results to euv_best_results.pkl")

    # 两种挑选判据
    print("\n═══ Best by max R_mean ═══")
    best_mean = pick_best(all_cands, 'R_mean')
    for w in WINDOWS:
        e = best_mean[w]
        print(f"  ±{w:>4}°  λ*={e['lam']:.1f} k*={e['k']}  "
              f"mean={e['R_mean']*100:.2f}% min={e['R_min']*100:.2f}% peak={e['R_peak']*100:.2f}%")

    print("\n═══ Best by max R_min ═══")
    best_min = pick_best(all_cands, 'R_min')
    for w in WINDOWS:
        e = best_min[w]
        print(f"  ±{w:>4}°  λ*={e['lam']:.1f} k*={e['k']}  "
              f"mean={e['R_mean']*100:.2f}% min={e['R_min']*100:.2f}% peak={e['R_peak']*100:.2f}%")

    print("\nPlotting...")
    plot_best(best_mean, per_evals, "max R_mean",  "fig_best_mean.png", fwhm)
    plot_best(best_min,  per_evals, "max R_min",   "fig_best_min.png",  fwhm)

    print(f"\nTotal wall time: {time.time()-t_total:.1f}s")


if __name__ == '__main__':
    main()
