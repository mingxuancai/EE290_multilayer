"""
Mo/Si EUV Aperiodic Multilayer — Parameter Sweeps

两个独立扫描：
  Sweep A: 固定 window=±10°（坡降区），扫 α ∈ [0, 1]  → α 如何切换 peak vs mean
  Sweep B: 固定 α=0.5，扫 window ∈ [0.5, 15]°         → 优势如何随角度窗口演化
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings, time
warnings.filterwarnings("ignore")

# ── 物理参数 ───────────────────────────────────────────────────────────────
LAMBDA_NM = 13.5
THETA0_DEG = 6.0
N_MO  = 0.921 + 1j * 0.0064
N_SI  = 0.999 + 1j * 0.0018
N_VAC = 1.0 + 0j
N_BILAYERS = 40

ALPHA_SWEEP_WINDOW = 10.0                                   # Sweep A 固定窗口
ALPHAS = np.array([0.0, 0.125, 0.25, 0.375, 0.5,
                   0.625, 0.75, 0.875, 1.0])
WINDOW_SWEEP_ALPHA = 0.5                                    # Sweep B 固定 α
WINDOWS = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 13.0, 15.0])


# ── TMM ────────────────────────────────────────────────────────────────────

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


def sweep(bilayers, theta_arr, top_mo=True):
    return np.array([R_stack(bilayers, t, top_mo) for t in theta_arr])


# ── 目标函数 ───────────────────────────────────────────────────────────────

def J(bilayers, window_deg, alpha, npts=9, top_mo=True):
    thetas = np.linspace(THETA0_DEG-window_deg, THETA0_DEG+window_deg, npts)
    R_arr = sweep(bilayers, thetas, top_mo)
    R_peak = R_stack(bilayers, THETA0_DEG, top_mo)
    return alpha*R_peak + (1-alpha)*np.mean(R_arr)


# ── 优化器 ─────────────────────────────────────────────────────────────────

def opt_periodic(N=N_BILAYERS):
    def neg_J(params):
        d, g = params
        bl = [(np.clip(d,4,14), np.clip(g,0.15,0.70))]*N
        return -R_stack(bl, THETA0_DEG)
    res = minimize(neg_J, [6.94, 0.39], method='Nelder-Mead',
                   options={'xatol':1e-5,'fatol':1e-8,'maxiter':800})
    d, g = np.clip(res.x[0],4,14), np.clip(res.x[1],0.15,0.70)
    return [(d, g)]*N, d, g


def greedy_grow(N, window_deg, alpha, d0, g0):
    bilayers = []
    for _ in range(N):
        def neg_J(params, _bl=bilayers):
            d, g = params
            full = _bl + [(np.clip(d,4,14), np.clip(g,0.15,0.70))]
            return -J(full, window_deg, alpha, npts=9)
        res = minimize(neg_J, [d0, g0], method='Nelder-Mead',
                       options={'xatol':1e-4,'fatol':1e-7,'maxiter':180,'adaptive':True})
        bilayers.append((np.clip(res.x[0],4,14), np.clip(res.x[1],0.15,0.70)))
    return bilayers


def coord_refine(bilayers, window_deg, alpha, passes=1):
    bl = list(bilayers)
    for _ in range(passes):
        for i in range(len(bl)):
            pre, suf = bl[:i], bl[i+1:]
            def neg_J(params, _pre=pre, _suf=suf):
                d, g = params
                full = _pre + [(np.clip(d,4,14), np.clip(g,0.15,0.70))] + _suf
                return -J(full, window_deg, alpha, npts=9)
            res = minimize(neg_J, list(bl[i]), method='Nelder-Mead',
                           options={'xatol':1e-4,'fatol':1e-7,'maxiter':120,'adaptive':True})
            bl[i] = (np.clip(res.x[0],4,14), np.clip(res.x[1],0.15,0.70))
    return bl


def evaluate(bl, window_deg):
    thetas = np.linspace(THETA0_DEG-window_deg, THETA0_DEG+window_deg, 81)
    R_arr = sweep(bl, thetas)
    return {
        'bl': bl,
        'R_peak': R_stack(bl, THETA0_DEG),
        'R_mean': float(np.mean(R_arr)),
        'R_min': float(np.min(R_arr)),
        'thetas': thetas,
        'R_arr': R_arr,
    }


# ── 主 ─────────────────────────────────────────────────────────────────────

def main():
    N = N_BILAYERS
    t_total = time.time()

    # periodic baseline
    print("Optimizing periodic baseline...")
    bl_per, d0, g0 = opt_periodic(N)
    R_per_peak = R_stack(bl_per, THETA0_DEG)
    print(f"  d={d0:.3f} nm, γ={g0:.3f}, R_peak={R_per_peak*100:.2f}%\n")

    # ── Sweep A: 扫 α，窗口固定 ──────────────────────────────────────────
    print(f"═══ Sweep A: α sweep (window=±{ALPHA_SWEEP_WINDOW}°) ═══")
    per_eval_A = evaluate(bl_per, ALPHA_SWEEP_WINDOW)

    sweep_A = {}
    for alpha in ALPHAS:
        t0 = time.time()
        bl = greedy_grow(N, ALPHA_SWEEP_WINDOW, alpha, d0, g0)
        bl = coord_refine(bl, ALPHA_SWEEP_WINDOW, alpha, passes=1)
        e = evaluate(bl, ALPHA_SWEEP_WINDOW)
        sweep_A[alpha] = e
        print(f"  α={alpha:.3f}  peak={e['R_peak']*100:.2f}%  "
              f"mean={e['R_mean']*100:.2f}%  min={e['R_min']*100:.2f}%  "
              f"({time.time()-t0:.1f}s)")

    # ── Sweep B: 扫 window，α 固定 ───────────────────────────────────────
    print(f"\n═══ Sweep B: window sweep (α={WINDOW_SWEEP_ALPHA}) ═══")
    sweep_B = {}
    for w in WINDOWS:
        t0 = time.time()
        bl = greedy_grow(N, w, WINDOW_SWEEP_ALPHA, d0, g0)
        bl = coord_refine(bl, w, WINDOW_SWEEP_ALPHA, passes=1)
        e_ap = evaluate(bl, w)
        e_per = evaluate(bl_per, w)
        sweep_B[w] = {'aperiodic': e_ap, 'periodic': e_per}
        print(f"  ±{w:>4}°  peak_ap={e_ap['R_peak']*100:.2f}%  "
              f"mean_ap={e_ap['R_mean']*100:.2f}%  "
              f"Δmean={(e_ap['R_mean']-e_per['R_mean'])*100:+.2f}%  "
              f"({time.time()-t0:.1f}s)")

    # Bragg FWHM
    thetas_full = np.linspace(0, 25, 500)
    R_full = sweep(bl_per, thetas_full)
    peak = R_full.max()
    above = thetas_full[R_full >= peak/2]
    fwhm = above[-1] - above[0]

    # ═══ PLOT A: α sweep ══════════════════════════════════════════════════
    print("\nPlotting Sweep A (α)...")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # A1: peak / mean / min vs α
    ax = axes[0]
    peaks  = np.array([sweep_A[a]['R_peak'] for a in ALPHAS])*100
    means  = np.array([sweep_A[a]['R_mean'] for a in ALPHAS])*100
    mins_  = np.array([sweep_A[a]['R_min']  for a in ALPHAS])*100
    ax.plot(ALPHAS, peaks, 'o-', color='#d73027', lw=2, ms=8, label='Aperiodic peak')
    ax.plot(ALPHAS, means, 's-', color='#1a9641', lw=2, ms=8, label='Aperiodic mean')
    ax.plot(ALPHAS, mins_, '^--', color='#756bb1', lw=1.6, ms=7, label='Aperiodic min')
    ax.axhline(per_eval_A['R_peak']*100, color='#d73027', ls=':', lw=1.4,
               label=f'Periodic peak {per_eval_A["R_peak"]*100:.1f}%')
    ax.axhline(per_eval_A['R_mean']*100, color='#1a9641', ls=':', lw=1.4,
               label=f'Periodic mean {per_eval_A["R_mean"]*100:.1f}%')
    ax.set_xlabel('α (weight on R_peak)', fontsize=11)
    ax.set_ylabel('Reflectivity (%)', fontsize=11)
    ax.set_title(f'α Sweep @ window=±{ALPHA_SWEEP_WINDOW}°\n'
                 f'(α=0 → maximize mean, α=1 → maximize peak)', fontsize=10)
    ax.legend(fontsize=8, loc='lower right'); ax.grid(alpha=0.3)

    # A2: Δ vs periodic
    ax = axes[1]
    dpeak = peaks - per_eval_A['R_peak']*100
    dmean = means - per_eval_A['R_mean']*100
    dmin  = mins_ - per_eval_A['R_min']*100
    ax.plot(ALPHAS, dpeak, 'o-', color='#d73027', lw=2, ms=8, label='Δ peak')
    ax.plot(ALPHAS, dmean, 's-', color='#1a9641', lw=2, ms=8, label='Δ mean')
    ax.plot(ALPHAS, dmin,  '^--', color='#756bb1', lw=1.6, ms=7, label='Δ min')
    ax.axhline(0, color='k', ls='--', lw=1.2)
    ax.set_xlabel('α', fontsize=11)
    ax.set_ylabel('Aperiodic − Periodic (% points)', fontsize=11)
    ax.set_title(f'Gain over Periodic vs α\n(window=±{ALPHA_SWEEP_WINDOW}°, slope region)', fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # A3: R(θ) overlay for select α
    ax = axes[2]
    thetas_w = per_eval_A['thetas']
    ax.plot(thetas_w, per_eval_A['R_arr']*100, 'k-', lw=2.8,
            label=f'Periodic (peak={per_eval_A["R_peak"]*100:.1f}%)', zorder=10)
    cmap = plt.cm.viridis
    for a in ALPHAS:
        e = sweep_A[a]
        ax.plot(thetas_w, e['R_arr']*100, lw=1.5, color=cmap(a),
                label=f'α={a:.2f}', alpha=0.85)
    ax.axvline(THETA0_DEG, color='gray', ls=':', lw=1)
    ax.set_xlabel('θ (°)', fontsize=11)
    ax.set_ylabel('R (%)', fontsize=11)
    ax.set_title(f'R(θ) curves: α rotates profile shape\n'
                 f'(±{ALPHA_SWEEP_WINDOW}° window)', fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc='lower center')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, None)

    plt.suptitle(f'Sweep A — α Parameter Sweep at Fixed Window ±{ALPHA_SWEEP_WINDOW}°  '
                 f'(Mo/Si N={N}, λ=13.5 nm, θ₀={THETA0_DEG}°)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_sweep_alpha.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig_sweep_alpha.png")

    # ═══ PLOT B: window sweep ═════════════════════════════════════════════
    print("\nPlotting Sweep B (window)...")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # B1: peak / mean vs window
    ax = axes[0]
    per_peak = np.array([sweep_B[w]['periodic']['R_peak'] for w in WINDOWS])*100
    per_mean = np.array([sweep_B[w]['periodic']['R_mean'] for w in WINDOWS])*100
    ap_peak  = np.array([sweep_B[w]['aperiodic']['R_peak'] for w in WINDOWS])*100
    ap_mean  = np.array([sweep_B[w]['aperiodic']['R_mean'] for w in WINDOWS])*100
    ax.plot(WINDOWS, per_peak, 'o:', color='#d73027', lw=1.6, ms=7, label='Periodic peak')
    ax.plot(WINDOWS, per_mean, 's:', color='#1a9641', lw=1.6, ms=7, label='Periodic mean')
    ax.plot(WINDOWS, ap_peak, 'o-', color='#d73027', lw=2.2, ms=8, label='Aperiodic peak')
    ax.plot(WINDOWS, ap_mean, 's-', color='#1a9641', lw=2.2, ms=8, label='Aperiodic mean')
    ax.axvline(fwhm/2, color='orange', ls='--', lw=1.4,
               label=f'Bragg half-width {fwhm/2:.1f}°')
    ax.set_xlabel('Window ±Δθ (°)', fontsize=11)
    ax.set_ylabel('Reflectivity (%)', fontsize=11)
    ax.set_title(f'Window Sweep @ α={WINDOW_SWEEP_ALPHA}\n'
                 f'peak/mean vs window', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # B2: Gain Δmean / Δpeak vs window
    ax = axes[1]
    dpeak = ap_peak - per_peak
    dmean = ap_mean - per_mean
    ax.plot(WINDOWS, dpeak, 'o-', color='#d73027', lw=2, ms=8, label='Δ peak')
    ax.plot(WINDOWS, dmean, 's-', color='#1a9641', lw=2, ms=8, label='Δ mean')
    for w, g in zip(WINDOWS, dmean):
        ax.annotate(f'{g:+.2f}', (w, g), fontsize=7.5,
                    textcoords='offset points', xytext=(0, 8), ha='center',
                    color='#1a9641')
    ax.axhline(0, color='k', lw=1.2, ls='--')
    ax.axvline(fwhm/2, color='orange', lw=1.4, ls='--',
               label=f'Bragg half-width {fwhm/2:.1f}°')
    ax.fill_betweenx([-5, 10], 0, fwhm/2, alpha=0.08, color='blue',
                     label='Plateau')
    ax.fill_betweenx([-5, 10], fwhm/2, 17, alpha=0.08, color='red',
                     label='Slope')
    ax.set_xlabel('Window ±Δθ (°)', fontsize=11)
    ax.set_ylabel('Aperiodic − Periodic (%)', fontsize=11)
    ax.set_title(f'Gain vs Window @ α={WINDOW_SWEEP_ALPHA}\n'
                 '(aperiodic advantage emerges past plateau)', fontsize=10)
    ax.set_xlim(0, 16)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # B3: R(θ) overlay for representative windows
    ax = axes[2]
    cmap = plt.cm.plasma
    for i, w in enumerate(WINDOWS):
        e_ap  = sweep_B[w]['aperiodic']
        e_per = sweep_B[w]['periodic']
        col = cmap(i/(len(WINDOWS)-1))
        ax.plot(e_ap['thetas'], e_ap['R_arr']*100, '-', lw=1.6, color=col,
                label=f'Ap ±{w}°')
        ax.plot(e_per['thetas'], e_per['R_arr']*100, ':', lw=1.2, color=col, alpha=0.6)
    ax.axvline(THETA0_DEG, color='k', ls=':', lw=1)
    ax.set_xlabel('θ (°)', fontsize=11)
    ax.set_ylabel('R (%)', fontsize=11)
    ax.set_title(f'R(θ) curves (solid=aperiodic, dotted=periodic)\n'
                 f'α={WINDOW_SWEEP_ALPHA}', fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc='lower center')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, None)

    plt.suptitle(f'Sweep B — Window Sweep at Fixed α={WINDOW_SWEEP_ALPHA}  '
                 f'(Mo/Si N={N}, λ=13.5 nm, θ₀={THETA0_DEG}°)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig_sweep_window.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig_sweep_window.png")

    print(f"\nTotal wall time: {time.time()-t_total:.1f}s")


if __name__ == '__main__':
    main()
