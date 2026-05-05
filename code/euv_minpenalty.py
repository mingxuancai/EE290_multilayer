"""
Mo/Si EUV Aperiodic — Objective: J = mean(R) + λ·min(R)
目标升级：用 λ 旋钮惩罚 dead-angle (保底型优化)。
反-greedy: multi-start (K=3 不同初值) + 2 轮 coord refine，取最优。
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

WINDOWS = [5.0, 10.0, 15.0]
LAMBDAS = [0.0, 0.5, 1.0, 2.0]   # min penalty 权重
K_STARTS = 3


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


# ── 新目标函数 J = mean + λ·min ─────────────────────────────────────────
def J_obj(bilayers, window_deg, lam, npts=13):
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
            return -J_obj(full, window_deg, lam, npts=13)
        res = minimize(neg, [d0, g0], method='Nelder-Mead',
                       options={'xatol':1e-4,'fatol':1e-7,'maxiter':180,'adaptive':True})
        bilayers.append((np.clip(res.x[0],4,14), np.clip(res.x[1],0.15,0.70)))
    return bilayers


def coord_refine(bilayers, window_deg, lam, passes=2):
    bl = list(bilayers)
    for _ in range(passes):
        for i in range(len(bl)):
            pre, suf = bl[:i], bl[i+1:]
            def neg(p, _pre=pre, _suf=suf):
                d, g = p
                full = _pre + [(np.clip(d,4,14), np.clip(g,0.15,0.70))] + _suf
                return -J_obj(full, window_deg, lam, npts=13)
            res = minimize(neg, list(bl[i]), method='Nelder-Mead',
                           options={'xatol':1e-4,'fatol':1e-7,'maxiter':130,'adaptive':True})
            bl[i] = (np.clip(res.x[0],4,14), np.clip(res.x[1],0.15,0.70))
    return bl


def multistart_optimize(window_deg, lam, d0_base, g0_base, K=K_STARTS):
    """K 个不同初值 greedy + refine，返回最优设计以及每次的 J。"""
    scales = np.linspace(0.94, 1.06, K)                       # 初值 d0 扰动
    best = None
    history = []
    for k, sc in enumerate(scales):
        d0 = np.clip(d0_base*sc, 4, 14)
        g0 = np.clip(g0_base*(1.0 + 0.02*(k - (K-1)/2)), 0.15, 0.70)
        bl = greedy_grow(window_deg, lam, d0, g0)
        bl = coord_refine(bl, window_deg, lam, passes=2)
        j_val = J_obj(bl, window_deg, lam, npts=81)
        history.append(j_val)
        if best is None or j_val > best[0]:
            best = (j_val, bl)
    return best[1], history


def evaluate(bl, window_deg):
    thetas = np.linspace(THETA0_DEG-window_deg, THETA0_DEG+window_deg, 161)
    R_arr = sweep_theta(bl, thetas)
    return {'bl': bl,
            'R_peak': R_stack(bl, THETA0_DEG),
            'R_mean': float(np.mean(R_arr)),
            'R_min':  float(np.min(R_arr)),
            'R_max':  float(np.max(R_arr)),
            'thetas': thetas, 'R_arr': R_arr}


def main():
    t_total = time.time()
    print("Optimizing periodic baseline...")
    bl_per, d0, g0 = opt_periodic()
    print(f"  d={d0:.3f} nm, γ={g0:.3f}, R_peak={R_stack(bl_per,THETA0_DEG)*100:.2f}%\n")

    thetas_full = np.linspace(0, 25, 500)
    R_full = sweep_theta(bl_per, thetas_full)
    above = thetas_full[R_full >= R_full.max()/2]
    fwhm = above[-1] - above[0]

    print(f"═══ Sweep: J = mean + λ·min, with multistart K={K_STARTS} ═══")
    results = {}  # results[(w, lam)] = eval dict
    multistart_logs = {}
    for w in WINDOWS:
        results[w] = {'per': evaluate(bl_per, w)}
        for lam in LAMBDAS:
            t0 = time.time()
            bl_best, history = multistart_optimize(w, lam, d0, g0)
            multistart_logs[(w, lam)] = history
            e = evaluate(bl_best, w)
            results[w][lam] = e
            pm = results[w]['per']['R_mean']*100
            pmin = results[w]['per']['R_min']*100
            print(f"  ±{w:>4}° λ={lam:.1f}  "
                  f"mean={e['R_mean']*100:.2f}%(Δ{e['R_mean']*100-pm:+.2f})  "
                  f"min={e['R_min']*100:.2f}%(Δ{e['R_min']*100-pmin:+.2f})  "
                  f"peak={e['R_peak']*100:.2f}%  "
                  f"K-hist={[f'{h*100:.1f}' for h in history]}  "
                  f"({time.time()-t0:.1f}s)")

    # ── PLOTS ─────────────────────────────────────────────────────────────
    print("\nPlotting...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    LCOLS = {0.0:'#1a9641', 0.5:'#f4a259', 1.0:'#e76f51', 2.0:'#6a3d9a'}

    # ── Row 1: R(θ) curves, one subplot per window
    for idx, w in enumerate(WINDOWS):
        ax = axes[0][idx]
        per = results[w]['per']
        ax.plot(per['thetas'], per['R_arr']*100, 'k-', lw=2.5,
                label=f'Periodic  mean={per["R_mean"]*100:.1f}%  min={per["R_min"]*100:.1f}%',
                zorder=10)
        for lam in LAMBDAS:
            e = results[w][lam]
            ax.plot(e['thetas'], e['R_arr']*100, '-', lw=1.8, color=LCOLS[lam],
                    label=f'λ={lam}  mean={e["R_mean"]*100:.1f}%  min={e["R_min"]*100:.1f}%')
        ax.axvline(THETA0_DEG, color='gray', ls=':', lw=1)
        ax.set_xlabel('θ (°)'); ax.set_ylabel('R (%)')
        in_plateau = w <= fwhm/2
        tag = "in plateau" if in_plateau else "on slope"
        ax.set_title(f'Window ±{w}°  [{tag}]', fontsize=10)
        ax.legend(fontsize=7.5, loc='lower center'); ax.grid(alpha=0.3)
        ax.set_ylim(0, None)

    # ── Row 2: Pareto & bar & multistart diagnostic

    # (1,0): mean vs min Pareto per window, λ traces curve
    ax = axes[1][0]
    for w in WINDOWS:
        per = results[w]['per']
        ax.scatter([per['R_min']*100], [per['R_mean']*100], marker='*', s=180,
                   color='k', edgecolors='w', linewidths=0.8, zorder=10,
                   label=f'Periodic ±{w}°')
        xs = [results[w][l]['R_min']*100 for l in LAMBDAS]
        ys = [results[w][l]['R_mean']*100 for l in LAMBDAS]
        ax.plot(xs, ys, 'o-', lw=1.8, ms=8,
                label=f'Aperiodic ±{w}° (λ trace)')
        for l, x, y in zip(LAMBDAS, xs, ys):
            ax.annotate(f'λ={l}', (x, y), fontsize=6.5,
                        textcoords='offset points', xytext=(5, 3))
    ax.set_xlabel('R_min (%)  (worst-case angle)')
    ax.set_ylabel('R_mean (%)')
    ax.set_title('Pareto: mean vs min (λ↑ pushes design toward higher min)', fontsize=10)
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

    # (1,1): Δmin and Δmean vs λ for each window
    ax = axes[1][1]
    for w in WINDOWS:
        dmean = [(results[w][l]['R_mean']-results[w]['per']['R_mean'])*100 for l in LAMBDAS]
        dmin  = [(results[w][l]['R_min']-results[w]['per']['R_min'])*100 for l in LAMBDAS]
        ax.plot(LAMBDAS, dmean, 'o-', lw=2, ms=8,
                label=f'Δmean ±{w}°')
        ax.plot(LAMBDAS, dmin,  's--', lw=1.6, ms=7, alpha=0.85,
                label=f'Δmin ±{w}°')
    ax.axhline(0, color='k', ls='--', lw=1)
    ax.set_xlabel('λ (weight on R_min in objective)')
    ax.set_ylabel('Aperiodic − Periodic (% points)')
    ax.set_title('Effect of λ: trades mean for min', fontsize=10)
    ax.legend(fontsize=7.5, ncol=2); ax.grid(alpha=0.3)

    # (1,2): Multistart diagnostic — show K runs per (w,λ)
    ax = axes[1][2]
    xlabels = []
    spread_min, spread_max, spread_best = [], [], []
    for w in WINDOWS:
        for lam in LAMBDAS:
            h = multistart_logs[(w, lam)]
            xlabels.append(f'±{int(w)}°\nλ={lam}')
            spread_min.append(min(h)*100)
            spread_max.append(max(h)*100)
            spread_best.append(max(h)*100)
    x = np.arange(len(xlabels))
    ax.vlines(x, spread_min, spread_max, color='gray', lw=3, alpha=0.6,
              label='K=3 multistart J-range')
    ax.scatter(x, spread_best, color='#d73027', s=60, zorder=10, label='Best')
    for i, (lo, hi) in enumerate(zip(spread_min, spread_max)):
        ax.annotate(f'{hi-lo:.2f}', (x[i], hi), fontsize=6,
                    textcoords='offset points', xytext=(0, 4), ha='center')
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=7, rotation=45)
    ax.set_ylabel('J value (%)')
    ax.set_title(f'Multistart K={K_STARTS}: J spread per config\n'
                 '(small spread → consistent; large → greedy stuck)', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    plt.suptitle(f'Mo/Si EUV Aperiodic — Objective: J = mean(R) + λ·min(R)\n'
                 f'Multistart K={K_STARTS} for greedy local-minimum avoidance   '
                 f'(N={N_BILAYERS}, λ=13.5 nm, θ₀={THETA0_DEG}°)',
                 fontsize=12, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('fig_minpenalty.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig_minpenalty.png")
    print(f"\nTotal wall time: {time.time()-t_total:.1f}s")


if __name__ == '__main__':
    main()
