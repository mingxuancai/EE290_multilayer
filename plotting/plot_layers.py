"""
Plot per-layer profiles of the best aperiodic designs from euv_best.py.
X-axis: bilayer index (1 = topmost, N = closest to Si substrate)
Y-axis: d_pair, γ, d_Mo, d_Si
Two figures: best_mean criterion, best_min criterion.
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path('/home/mingxuan/project/Litho')
with open(ROOT / 'euv_best_results.pkl', 'rb') as f:
    data = pickle.load(f)

all_cands = data['all_cands']
per_evals = data['per_evals']
fwhm      = data['fwhm']
d0_per    = data['d0']
g0_per    = data['g0']

WINDOWS = sorted({w for (w, _, _) in all_cands.keys()})
LAMBDAS = sorted({l for (_, l, _) in all_cands.keys()})
K       = 1 + max(k for (_, _, k) in all_cands.keys())

def pick_best(metric):
    best = {}
    for w in WINDOWS:
        cands = [all_cands[(w, l, k)] for l in LAMBDAS for k in range(K)]
        best[w] = max(cands, key=lambda e: e[metric])
    return best


def plot_profiles(best, title, outpath):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    cmap = plt.cm.plasma
    N = None
    for i, w in enumerate(WINDOWS):
        bl = best[w]['bl']
        N = len(bl)
        idx = np.arange(1, N+1)
        d_pair = np.array([b[0] for b in bl])
        gamma  = np.array([b[1] for b in bl])
        d_Mo   = d_pair * gamma
        d_Si   = d_pair * (1 - gamma)
        col = cmap(i/(len(WINDOWS)-1))
        lab = f'±{w}°'

        axes[0][0].plot(idx, d_pair, 'o-', color=col, lw=1.4, ms=4, label=lab)
        axes[0][1].plot(idx, gamma,  's-', color=col, lw=1.4, ms=4, label=lab)
        axes[1][0].plot(idx, d_Mo,   '^-', color=col, lw=1.4, ms=4, label=lab)
        axes[1][1].plot(idx, d_Si,   'v-', color=col, lw=1.4, ms=4, label=lab)

    # periodic reference lines
    axes[0][0].axhline(d0_per, color='k', ls=':', lw=1.5, label=f'Periodic {d0_per:.2f} nm')
    axes[0][1].axhline(g0_per, color='k', ls=':', lw=1.5, label=f'Periodic {g0_per:.3f}')
    axes[1][0].axhline(d0_per*g0_per, color='k', ls=':', lw=1.5,
                       label=f'Periodic {d0_per*g0_per:.2f} nm')
    axes[1][1].axhline(d0_per*(1-g0_per), color='k', ls=':', lw=1.5,
                       label=f'Periodic {d0_per*(1-g0_per):.2f} nm')

    titles = [
        f'Bilayer thickness d_pair (nm)\nbounds [4, 14] nm',
        f'Mo fraction γ = d_Mo / d_pair\nbounds [0.15, 0.70]',
        f'Mo layer thickness d_Mo = d_pair·γ (nm)',
        f'Si layer thickness d_Si = d_pair·(1−γ) (nm)',
    ]
    ylabels = ['d_pair (nm)', 'γ', 'd_Mo (nm)', 'd_Si (nm)']
    for ax, t, yl in zip(axes.flatten(), titles, ylabels):
        ax.set_xlabel(f'Bilayer index  (1 = top / vacuum-side,  {N} = substrate-side)')
        ax.set_ylabel(yl)
        ax.set_title(t, fontsize=10)
        ax.legend(fontsize=7, ncol=2, loc='best')
        ax.grid(alpha=0.3)
        ax.set_xlim(0.5, N+0.5)

    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {outpath}")


# generate two figures
best_mean = pick_best('R_mean')
best_min  = pick_best('R_min')

plot_profiles(best_mean,
              f'Per-Layer Profiles — Best Designs by max R_mean\n'
              f'(Mo/Si N=40, λ=13.5 nm, θ₀=6°, window ±Δθ ∈ {WINDOWS})',
              ROOT / 'fig_layers_best_mean.png')

plot_profiles(best_min,
              f'Per-Layer Profiles — Best Designs by max R_min\n'
              f'(Mo/Si N=40, λ=13.5 nm, θ₀=6°, window ±Δθ ∈ {WINDOWS})',
              ROOT / 'fig_layers_best_min.png')

# numerical summary
print("\n── Summary: best_mean designs — spread of d_pair per window ──")
print(f"{'Window':>8} {'λ*':>5} {'min(d_pair)':>12} {'max(d_pair)':>12} "
      f"{'mean(d)':>10} {'std(d)':>8} {'min(γ)':>8} {'max(γ)':>8}")
for w in WINDOWS:
    bl = best_mean[w]['bl']
    d = np.array([b[0] for b in bl])
    g = np.array([b[1] for b in bl])
    print(f"  ±{w:>4}°  {best_mean[w]['lam']:>4.1f}  "
          f"{d.min():>11.3f}  {d.max():>11.3f}  "
          f"{d.mean():>9.3f}  {d.std():>7.3f}  "
          f"{g.min():>7.3f}  {g.max():>7.3f}")

print("\n── Summary: best_min designs — spread of d_pair per window ──")
print(f"{'Window':>8} {'λ*':>5} {'min(d_pair)':>12} {'max(d_pair)':>12} "
      f"{'mean(d)':>10} {'std(d)':>8} {'min(γ)':>8} {'max(γ)':>8}")
for w in WINDOWS:
    bl = best_min[w]['bl']
    d = np.array([b[0] for b in bl])
    g = np.array([b[1] for b in bl])
    print(f"  ±{w:>4}°  {best_min[w]['lam']:>4.1f}  "
          f"{d.min():>11.3f}  {d.max():>11.3f}  "
          f"{d.mean():>9.3f}  {d.std():>7.3f}  "
          f"{g.min():>7.3f}  {g.max():>7.3f}")
