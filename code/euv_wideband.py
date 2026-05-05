"""
Mo/Si EUV Aperiodic Multilayer — Wide-Angle Study
修正版：FWHM~12°，需要 ±8-15° 才能看出 aperiodic 优势

研究设计：
- 1 个优化好的 periodic 结构，评估在所有窗口上的表现
- 对每个窗口独立优化 aperiodic（α∈{0, 0.5, 1}）
- 图表清晰展示"平台区（≤6°）无优势"vs"坡降区（>8°）有优势"
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# ── 物理参数 ───────────────────────────────────────────────────────────────
LAMBDA_NM = 13.5
THETA0_DEG = 6.0
N_MO  = 0.921 + 1j * 0.0064
N_SI  = 0.999 + 1j * 0.0018
N_VAC = 1.0 + 0j
N_BILAYERS = 40

# 研究的角度窗口（± 度）
WINDOWS = [1.0, 3.0, 6.0, 10.0, 15.0]

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
    sin0 = np.sin(np.deg2rad(theta_deg))   # n_vac=1
    kz = k0 * np.sqrt(n_list**2 - sin0**2 + 0j)
    Rs = abs(_rec_r(n_list, kz, d_list, 's'))**2
    Rp = abs(_rec_r(n_list, kz, d_list, 'p'))**2
    return 0.5*(float(Rs)+float(Rp))


def sweep(bilayers, theta_arr, top_mo=True):
    return np.array([R_stack(bilayers, t, top_mo) for t in theta_arr])


# ── 目标函数 ───────────────────────────────────────────────────────────────

def J(bilayers, window_deg, alpha, npts=11, top_mo=True):
    """J = α·R(θ₀) + (1-α)·mean(R over θ₀±window)"""
    thetas = np.linspace(THETA0_DEG-window_deg, THETA0_DEG+window_deg, npts)
    R_arr = sweep(bilayers, thetas, top_mo)
    R_peak = R_stack(bilayers, THETA0_DEG, top_mo)
    return alpha*R_peak + (1-alpha)*np.mean(R_arr)


# ── 优化器 ─────────────────────────────────────────────────────────────────

def opt_periodic(N=N_BILAYERS):
    """优化 periodic 结构，最大化 peak reflectivity。"""
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
    for i in range(N):
        def neg_J(params, _bl=bilayers):
            d, g = params
            full = _bl + [(np.clip(d,4,14), np.clip(g,0.15,0.70))]
            return -J(full, window_deg, alpha, npts=11)
        res = minimize(neg_J, [d0, g0], method='Nelder-Mead',
                       options={'xatol':1e-4,'fatol':1e-7,'maxiter':200,'adaptive':True})
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
                return -J(full, window_deg, alpha, npts=11)
            res = minimize(neg_J, list(bl[i]), method='Nelder-Mead',
                           options={'xatol':1e-4,'fatol':1e-7,'maxiter':150,'adaptive':True})
            bl[i] = (np.clip(res.x[0],4,14), np.clip(res.x[1],0.15,0.70))
    return bl


# ── 主函数 ─────────────────────────────────────────────────────────────────

def main():
    N = N_BILAYERS
    alphas = [0.0, 0.5, 1.0]
    top_mo = True

    # ── Step 1: 优化 periodic 结构 ─────────────────────────────────────
    print("Step 1: Optimizing periodic baseline...")
    bl_per, d0, g0 = opt_periodic(N)
    R_per_peak = R_stack(bl_per, THETA0_DEG)

    # 计算 periodic 的全角度响应（宽扫）
    thetas_full = np.linspace(0, 25, 500)
    R_full = sweep(bl_per, thetas_full)
    peak = R_full.max()
    above_half = thetas_full[R_full >= peak/2]
    fwhm = above_half[-1] - above_half[0]
    print(f"  Periodic: d_pair={d0:.3f} nm, γ={g0:.3f}")
    print(f"  R_peak={R_per_peak*100:.2f}%, FWHM={fwhm:.2f}°, half-width={fwhm/2:.2f}°")
    print(f"  → ±{fwhm/2:.1f}° plateau: windows ≤{fwhm/2:.0f}° have NO aperiodic advantage")

    # ── Step 2: 对每个窗口优化 aperiodic ──────────────────────────────
    results = {}
    for window in WINDOWS:
        print(f"\nWindow ±{window}°:")
        results[window] = {}

        # 评估 periodic 在此窗口的性能（无需重新优化）
        thetas_w = np.linspace(THETA0_DEG-window, THETA0_DEG+window, 81)
        R_per_arr = sweep(bl_per, thetas_w)
        results[window]['periodic'] = {
            'bl': bl_per,
            'R_peak': R_per_peak,
            'R_mean': np.mean(R_per_arr),
            'R_arr': R_per_arr,
            'thetas': thetas_w,
        }

        # 优化 aperiodic（每个 alpha）
        results[window]['aperiodic'] = {}
        for alpha in alphas:
            print(f"  α={alpha:.1f}...", end=" ", flush=True)
            bl_g = greedy_grow(N, window, alpha, d0, g0)
            bl_r = coord_refine(bl_g, window, alpha, passes=1)
            R_ap_arr = sweep(bl_r, thetas_w)
            R_ap_peak = R_stack(bl_r, THETA0_DEG)
            R_ap_mean = np.mean(R_ap_arr)
            print(f"peak={R_ap_peak*100:.2f}%, mean={R_ap_mean*100:.2f}%  "
                  f"(Δpeak={( R_ap_peak-R_per_peak)*100:+.2f}%  "
                  f"Δmean={(R_ap_mean-results[window]['periodic']['R_mean'])*100:+.2f}%)")
            results[window]['aperiodic'][alpha] = {
                'bl': bl_r,
                'R_peak': R_ap_peak,
                'R_mean': R_ap_mean,
                'R_arr': R_ap_arr,
            }

    # ── PLOTS ──────────────────────────────────────────────────────────────
    print("\nGenerating figures...")

    WCOLS = {1.0:'#4393c3', 3.0:'#92c5de', 6.0:'#fdbf6f',
             10.0:'#d73027', 15.0:'#762a83'}
    ACOLS = {0.0:'#1a9641', 0.5:'#fd8d3c', 1.0:'#756bb1'}

    # ── Fig 0: 带宽物理示意 ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(thetas_full, R_full*100, lw=2.5, color='#2c7bb6', label='Periodic Mo/Si N=40')
    ax.axhline(R_per_peak*100/2, color='gray', ls=':', lw=1.2, label='FWHM level')
    ax.axvspan(above_half[0], above_half[-1], alpha=0.10, color='#2c7bb6',
               label=f'FWHM = {fwhm:.1f}°')
    for w, col in zip(WINDOWS, ['#d0e9f7','#a6cee3','#fdbf6f','#e31a1c','#6a3d9a']):
        ax.axvspan(THETA0_DEG-w, THETA0_DEG+w, alpha=0.15, color=col,
                   label=f'±{w}°' if w in [1,6,15] else f'±{w}°')
    ax.axvline(THETA0_DEG, color='k', ls='--', lw=1.2)
    ax.set_xlabel('Angle from normal (°)')
    ax.set_ylabel('Reflectivity (%)')
    ax.set_title(f'Mo/Si Periodic (N={N}) Angular Profile\n'
                 f'FWHM≈{fwhm:.1f}° → windows ≤ {fwhm/2:.0f}° are in the "plateau"', fontsize=10)
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(0, 25); ax.grid(alpha=0.3)

    # 坡降区放大
    ax2 = axes[1]
    thetas_zoom = np.linspace(THETA0_DEG-16, THETA0_DEG+16, 400)
    R_zoom = sweep(bl_per, thetas_zoom)
    ax2.plot(thetas_zoom, R_zoom*100, lw=2.5, color='#2c7bb6', label='Periodic')
    for w in WINDOWS:
        col = WCOLS[w]
        ax2.axvspan(THETA0_DEG-w, THETA0_DEG+w, alpha=0.20, color=col, label=f'±{w}°')
    ax2.axvline(THETA0_DEG, color='k', ls='--', lw=1)
    ax2.set_xlabel('Angle from normal (°)')
    ax2.set_ylabel('Reflectivity (%)')
    ax2.set_title('Zoom: θ₀ ± 16°\n'
                  '(±1°,3° in plateau; ±6° at slope; ±10°,15° in descent)', fontsize=10)
    ax2.legend(fontsize=7.5); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig0_bandwidth.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig0_bandwidth.png")

    # ── Fig 1: R(θ) 曲线，每个窗口一个子图 ─────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for idx, window in enumerate(WINDOWS):
        ax = axes[idx]
        thetas_w = results[window]['periodic']['thetas']
        R_per = results[window]['periodic']['R_arr']
        ax.plot(thetas_w, R_per*100, lw=2.5, color='#2c7bb6', zorder=10,
                label=f"Periodic  peak={results[window]['periodic']['R_peak']*100:.1f}%"
                      f"  mean={results[window]['periodic']['R_mean']*100:.1f}%")
        for alpha in alphas:
            d = results[window]['aperiodic'][alpha]
            ax.plot(thetas_w, d['R_arr']*100, lw=1.8, color=ACOLS[alpha],
                    ls=['-','--',':'][alphas.index(alpha)],
                    label=f"Ap α={alpha}  peak={d['R_peak']*100:.1f}%  mean={d['R_mean']*100:.1f}%")
        ax.axvline(THETA0_DEG, color='k', ls=':', lw=0.8)
        ax.axvspan(THETA0_DEG-window, THETA0_DEG+window, alpha=0.07, color='gray')
        in_plateau = window <= fwhm/2
        tag = "PLATEAU — no advantage" if in_plateau else "SLOPE — aperiodic can gain"
        ax.set_title(f'±{window}° window  [{tag}]', fontsize=9)
        ax.set_xlabel('θ (°)'); ax.set_ylabel('R (%)')
        ax.legend(fontsize=6.5, loc='lower center'); ax.grid(alpha=0.3); ax.set_ylim(0, 100)

    axes[-1].set_visible(False)
    plt.suptitle(f'Periodic vs Aperiodic R(θ) at Different Angular Windows\n'
                 f'Mo/Si N={N}, λ=13.5 nm, θ₀={THETA0_DEG}°', fontsize=11)
    plt.tight_layout()
    plt.savefig('fig1_Rtheta_windows.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig1_Rtheta_windows.png")

    # ── Fig 2: Pareto front 随窗口演变 ──────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.set_title('Pareto Front: Peak R vs Mean R for Each Window\n'
                 '(★ = periodic, curves = aperiodic Pareto)', fontsize=10)
    for window in WINDOWS:
        col = WCOLS[window]
        per = results[window]['periodic']
        ap_peaks = [results[window]['aperiodic'][a]['R_peak']*100 for a in alphas]
        ap_means = [results[window]['aperiodic'][a]['R_mean']*100 for a in alphas]
        ax.scatter([per['R_mean']*100], [per['R_peak']*100],
                   s=130, marker='*', color=col, zorder=10,
                   edgecolors='k', linewidths=0.8, label=f'Per ±{window}°')
        ax.plot(ap_means, ap_peaks, 'o-', color=col, ms=6, lw=1.5,
                alpha=0.85, label=f'Ap ±{window}°')
        for a, x, y in zip(alphas, ap_means, ap_peaks):
            if a in [0.0, 1.0]:
                ax.annotate(f'α={a}', (x, y), fontsize=5.5,
                             textcoords='offset points', xytext=(3, 2))
    ax.set_xlabel('Mean R over ±window (%)')
    ax.set_ylabel('Peak R at θ₀ (%)')
    ax.legend(fontsize=6.5, ncol=2); ax.grid(alpha=0.3)

    # Fig 2 右：ΔR_mean vs window
    ax2 = axes[1]
    ax2.set_title('Aperiodic Gain over Periodic: ΔR_mean vs Window\n'
                  '(positive = aperiodic wins on angular average)', fontsize=10)
    for alpha in alphas:
        gains = [(results[w]['aperiodic'][alpha]['R_mean'] -
                  results[w]['periodic']['R_mean'])*100 for w in WINDOWS]
        ax2.plot(WINDOWS, gains, 'o-', color=ACOLS[alpha], lw=2, ms=8,
                 label=f'α={alpha}')
        for w, g in zip(WINDOWS, gains):
            ax2.annotate(f'{g:+.2f}%', (w, g), fontsize=7,
                         textcoords='offset points', xytext=(0, 8), ha='center')
    ax2.axhline(0, color='k', lw=1.2, ls='--')
    ax2.axvline(fwhm/2, color='orange', lw=1.5, ls='--',
                label=f'FWHM half-width ({fwhm/2:.1f}°)')
    ax2.fill_betweenx([-5, 10], 0, fwhm/2, alpha=0.07, color='blue',
                      label='Plateau region')
    ax2.fill_betweenx([-5, 10], fwhm/2, 20, alpha=0.07, color='red',
                      label='Slope region')
    ax2.set_xlabel('Window ±Δθ (degrees)')
    ax2.set_ylabel('ΔR_mean aperiodic − periodic (%)')
    ax2.set_xlim(0, 17)
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig2_pareto_gain.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig2_pareto_gain.png")

    # ── Fig 3: 厚度分布（±6° 和 ±15° 两个代表性窗口，α=0）──────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for row, window in enumerate([6.0, 15.0]):
        for col_i, alpha in enumerate([0.0, 0.5]):
            ax = axes[row][col_i]
            bl = results[window]['aperiodic'][alpha]['bl']
            d_pairs = [d for d, g in bl]
            gammas  = [g for d, g in bl]
            idx = np.arange(1, len(bl)+1)
            ax2_ = ax.twinx()
            ax.bar(idx, d_pairs, color='#4292c6', alpha=0.75, label='d_pair (nm)')
            ax.axhline(d0, color='navy', ls='--', lw=1.2,
                       label=f'Periodic d={d0:.2f} nm')
            ax2_.plot(idx, gammas, 'r.-', ms=5, lw=1.2, label='γ')
            ax2_.axhline(g0, color='darkred', ls=':', lw=1)
            ax.set_title(f'±{window}° window, α={alpha}\n'
                         f"peak={results[window]['aperiodic'][alpha]['R_peak']*100:.1f}%"
                         f"  mean={results[window]['aperiodic'][alpha]['R_mean']*100:.1f}%",
                         fontsize=9)
            ax.set_xlabel('Bilayer index')
            ax.set_ylabel('d_pair (nm)', color='#4292c6')
            ax2_.set_ylabel('γ = d_Mo/d_pair', color='r')
            ax.set_ylim(0, 15); ax2_.set_ylim(0.1, 0.8)
            ax.legend(fontsize=7, loc='upper left')
    plt.suptitle(f'Bilayer Thickness Profiles: Transition from ±6° to ±15° Window\n'
                 f'(chirped/graded structure emerges as window grows)', fontsize=11)
    plt.tight_layout()
    plt.savefig('fig3_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig3_profiles.png")

    # ── Fig 4: 大总结图 ───────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.33)

    # (0,0): bandwidth overview
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.plot(thetas_zoom, R_zoom*100, lw=2, color='#2c7bb6')
    for w in WINDOWS:
        ax00.axvspan(THETA0_DEG-w, THETA0_DEG+w, alpha=0.18, color=WCOLS[w],
                     label=f'±{w}°')
    ax00.axvline(THETA0_DEG, color='k', ls='--', lw=1)
    ax00.set_title(f'Periodic FWHM≈{fwhm:.1f}°\n(windows colored)', fontsize=9)
    ax00.set_xlabel('θ (°)'); ax00.set_ylabel('R (%)'); ax00.legend(fontsize=6.5)
    ax00.grid(alpha=0.3)

    # (0,1): ±3° (in plateau)
    w3 = 3.0
    ax01 = fig.add_subplot(gs[0, 1])
    thw = results[w3]['periodic']['thetas']
    ax01.plot(thw, results[w3]['periodic']['R_arr']*100, lw=2.5, color='#2c7bb6',
              label=f"Periodic mean={results[w3]['periodic']['R_mean']*100:.1f}%")
    for alpha in alphas:
        d = results[w3]['aperiodic'][alpha]
        ax01.plot(thw, d['R_arr']*100, lw=1.8, color=ACOLS[alpha],
                  ls=['-','--',':'][alphas.index(alpha)],
                  label=f"Ap α={alpha} mean={d['R_mean']*100:.1f}%")
    ax01.set_title(f'±{w3}° (IN plateau → ~no gain)', fontsize=9)
    ax01.set_xlabel('θ (°)'); ax01.set_ylabel('R (%)'); ax01.set_ylim(0,100)
    ax01.legend(fontsize=6.5); ax01.grid(alpha=0.3)

    # (0,2): ±10° (on slope)
    w10 = 10.0
    ax02 = fig.add_subplot(gs[0, 2])
    thw = results[w10]['periodic']['thetas']
    ax02.plot(thw, results[w10]['periodic']['R_arr']*100, lw=2.5, color='#2c7bb6',
              label=f"Periodic mean={results[w10]['periodic']['R_mean']*100:.1f}%")
    for alpha in alphas:
        d = results[w10]['aperiodic'][alpha]
        ax02.plot(thw, d['R_arr']*100, lw=1.8, color=ACOLS[alpha],
                  ls=['-','--',':'][alphas.index(alpha)],
                  label=f"Ap α={alpha} mean={d['R_mean']*100:.1f}%")
    ax02.set_title(f'±{w10}° (ON slope → aperiodic gains)', fontsize=9)
    ax02.set_xlabel('θ (°)'); ax02.set_ylabel('R (%)'); ax02.set_ylim(0,100)
    ax02.legend(fontsize=6.5); ax02.grid(alpha=0.3)

    # (1,0-1): Gain vs window
    ax10 = fig.add_subplot(gs[1, 0:2])
    for alpha in alphas:
        gains = [(results[w]['aperiodic'][alpha]['R_mean'] -
                  results[w]['periodic']['R_mean'])*100 for w in WINDOWS]
        ax10.plot(WINDOWS, gains, 'o-', color=ACOLS[alpha], lw=2.5, ms=9,
                  label=f'α={alpha}')
        for w, g in zip(WINDOWS, gains):
            ax10.annotate(f'{g:+.2f}%', (w, g), fontsize=8,
                          textcoords='offset points', xytext=(0, 9), ha='center')
    ax10.axhline(0, color='k', lw=1.5, ls='--', label='Break-even')
    ax10.axvline(fwhm/2, color='#e6851e', lw=2, ls='--',
                 label=f'Bragg half-width ({fwhm/2:.1f}°)')
    ax10.fill_betweenx([-5, 12], 0, fwhm/2, alpha=0.07, color='blue')
    ax10.fill_betweenx([-5, 12], fwhm/2, 17, alpha=0.07, color='red')
    ax10.text(fwhm/4, 10, 'Plateau\n(no gain)', ha='center', fontsize=9, color='blue')
    ax10.text(fwhm/2+3, 10, 'Slope\n(aperiodic gains)', ha='center', fontsize=9, color='red')
    ax10.set_xlabel('Angular Window ±Δθ (degrees)', fontsize=10)
    ax10.set_ylabel('ΔR_mean: Aperiodic − Periodic (%)', fontsize=10)
    ax10.set_title('When does Aperiodic Beat Periodic? (Angular Mean Reflectivity)', fontsize=10)
    ax10.legend(fontsize=8); ax10.grid(alpha=0.3)
    ax10.set_xlim(0, 17)

    # (1,2): Pareto front for ±10° and ±15°
    ax12 = fig.add_subplot(gs[1, 2])
    ax12.set_title('Pareto Front at ±10° and ±15°\n(aperiodic separates from periodic ★)',
                   fontsize=9)
    for window in [10.0, 15.0]:
        col = WCOLS[window]
        per = results[window]['periodic']
        ap_peaks = [results[window]['aperiodic'][a]['R_peak']*100 for a in alphas]
        ap_means = [results[window]['aperiodic'][a]['R_mean']*100 for a in alphas]
        ax12.scatter([per['R_mean']*100], [per['R_peak']*100],
                     s=160, marker='*', color=col, zorder=10,
                     edgecolors='k', linewidths=0.8, label=f'Periodic ±{window}°')
        ax12.plot(ap_means, ap_peaks, 'o-', color=col, ms=8, lw=2,
                  label=f'Aperiodic ±{window}°')
    ax12.set_xlabel('Mean R (%)'); ax12.set_ylabel('Peak R (%)')
    ax12.legend(fontsize=8); ax12.grid(alpha=0.3)

    plt.suptitle(f'Mo/Si EUV Multilayer: When Does Aperiodic Help?\n'
                 f'λ=13.5 nm, N={N}, θ₀={THETA0_DEG}° from normal',
                 fontsize=12, fontweight='bold')
    plt.savefig('fig4_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig4_summary.png")

    # ── 数值汇总 ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"SUMMARY (N={N}, λ=13.5 nm, θ₀={THETA0_DEG}°)")
    print(f"Periodic: d={d0:.3f} nm, γ={g0:.3f}, R_peak={R_per_peak*100:.2f}%")
    print(f"Bragg peak half-width (FWHM/2) = {fwhm/2:.1f}°")
    print("="*70)
    print(f"{'Window':>7} | {'Per_mean%':>10} | {'Ap0_mean%':>10} | {'Ap05_mean%':>11}"
          f" | {'Ap1_mean%':>10} | verdict")
    print("-"*70)
    for w in WINDOWS:
        pm  = results[w]['periodic']['R_mean']*100
        am0 = results[w]['aperiodic'][0.0]['R_mean']*100
        am5 = results[w]['aperiodic'][0.5]['R_mean']*100
        am1 = results[w]['aperiodic'][1.0]['R_mean']*100
        in_plateau = w <= fwhm/2
        best_gain = max(am0-pm, am5-pm, am1-pm)
        verdict = ("no gain (plateau)" if in_plateau and best_gain < 0.05
                   else f"+{best_gain:.2f}% gain")
        print(f"  ±{w:>4}° | {pm:10.2f} | {am0:10.2f} | {am5:11.2f}"
              f" | {am1:10.2f} | {verdict}")
    print()


if __name__ == '__main__':
    main()
