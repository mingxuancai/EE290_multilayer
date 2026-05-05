"""
Mirror of plot_pol_best_mean.py for the avg-opt (unpolarized) results.
Best design per window selected by max R_avg_mean.

Outputs:
  fig_avg_periodic_mean.png  — periodic R_avg vs window (reference)
  fig_avg_Rmean.png          — R_avg_mean: aperiodic vs periodic
  fig_avg_Rpeak.png          — R_avg @ θ₀: aperiodic vs periodic
  fig_avg_Rmin.png           — R_avg_min: aperiodic vs periodic
  fig_avg_layers.png         — per-layer (d_pair / γ / d_Mo / d_Si) profiles
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path('/home/mingxuan/project/Litho')

LAMBDA_NM = 13.5
THETA0_DEG = 6.0
N_MO  = 0.921 + 1j * 0.0064
N_SI  = 0.999 + 1j * 0.0018
N_VAC = 1.0 + 0j


def build_stack(bilayers):
    d = [0.0]; n = [N_VAC]
    for d_pair, gamma in bilayers:
        d += [d_pair*gamma, d_pair*(1-gamma)]; n += [N_MO, N_SI]
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

def R_avg(bl, theta_deg):
    d_list, n_list = build_stack(bl)
    k0 = 2*np.pi / LAMBDA_NM
    sin0 = np.sin(np.deg2rad(theta_deg))
    kz = k0 * np.sqrt(n_list**2 - sin0**2 + 0j)
    Rs = abs(_rec_r(n_list, kz, d_list, 's'))**2
    Rp = abs(_rec_r(n_list, kz, d_list, 'p'))**2
    return 0.5*(float(Rs)+float(Rp))


with open(ROOT / 'euv_pol_results.pkl', 'rb') as f:
    data = pickle.load(f)

cands_unpol = data['cands_unpol']
bl_per_avg  = data['bl_per_avg']

WINDOWS = sorted({w for (w, _, _) in cands_unpol.keys()})
LAMBDAS = sorted({l for (_, l, _) in cands_unpol.keys()})
K       = 1 + max(k for (_, _, k) in cands_unpol.keys())

d0_per = bl_per_avg[0][0]
g0_per = bl_per_avg[0][1]

# Best by max R_avg_mean
best_mean = {}
for w in WINDOWS:
    cands = [cands_unpol[(w, l, k)] for l in LAMBDAS for k in range(K)]
    best_mean[w] = max(cands, key=lambda e: e['mean_av'])


def per_eval_avg(w):
    th = np.linspace(THETA0_DEG-w, THETA0_DEG+w, 161)
    R_arr = np.array([R_avg(bl_per_avg, t) for t in th])
    return {'mean': float(R_arr.mean()),
            'min':  float(R_arr.min()),
            'peak': R_avg(bl_per_avg, THETA0_DEG)}
per = {w: per_eval_avg(w) for w in WINDOWS}

th_full = np.linspace(0, 25, 500)
R_full = np.array([R_avg(bl_per_avg, t) for t in th_full])
above = th_full[R_full >= R_full.max()/2]
fwhm = above[-1] - above[0]
print(f"Periodic AVG FWHM = {fwhm:.2f}°, half-width = {fwhm/2:.2f}°")
print(f"Periodic AVG R_peak = {R_avg(bl_per_avg, THETA0_DEG)*100:.2f}%\n")


def style_axes(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=12); ax.grid(alpha=0.3)


# ── Fig 1 — Periodic R_avg_mean vs window ────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(WINDOWS, [per[w]['mean']*100 for w in WINDOWS],
        's-', color='#1a4480', lw=2.5, ms=10, label='Periodic R_avg mean')
ax.axvline(fwhm/2, color='orange', ls='--', lw=1.5,
           label=f'Bragg half-width {fwhm/2:.1f}°')
style_axes(ax, 'Window ±Δθ (°)', 'R_avg_mean (%)',
           'Periodic Multilayer — Mean R_avg over ±Window')
plt.tight_layout()
plt.savefig(ROOT / 'fig_avg_periodic_mean.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig_avg_periodic_mean.png")

# ── Fig 2 — R_avg_mean vs window ────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(WINDOWS, [per[w]['mean']*100         for w in WINDOWS],
        's:', color='#1a9641', lw=1.8, ms=8, label='Periodic')
ax.plot(WINDOWS, [best_mean[w]['mean_av']*100 for w in WINDOWS],
        's-', color='#1a9641', lw=2.6, ms=10, label='Aperiodic (avg-opt, best mean)')
ax.axvline(fwhm/2, color='orange', ls='--', lw=1.5,
           label=f'Bragg half-width {fwhm/2:.1f}°')
style_axes(ax, 'Window ±Δθ (°)', 'R_avg_mean (%)',
           'Mean R_avg over ±Window')
plt.tight_layout()
plt.savefig(ROOT / 'fig_avg_Rmean.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig_avg_Rmean.png")

# ── Fig 3 — R_avg_peak vs window ────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
ap_peak = [R_avg(best_mean[w]['bl'], THETA0_DEG)*100 for w in WINDOWS]
ax.plot(WINDOWS, [per[w]['peak']*100 for w in WINDOWS],
        'o:', color='#d73027', lw=1.8, ms=8, label='Periodic')
ax.plot(WINDOWS, ap_peak,
        'o-', color='#d73027', lw=2.6, ms=10, label='Aperiodic (avg-opt, best mean)')
ax.axvline(fwhm/2, color='orange', ls='--', lw=1.5,
           label=f'Bragg half-width {fwhm/2:.1f}°')
style_axes(ax, 'Window ±Δθ (°)', 'R_avg_peak @ θ₀=6° (%)',
           'Peak R_avg at θ₀ vs ±Window')
plt.tight_layout()
plt.savefig(ROOT / 'fig_avg_Rpeak.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig_avg_Rpeak.png")

# ── Fig 4 — R_avg_min vs window ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(WINDOWS, [per[w]['min']*100         for w in WINDOWS],
        '^:', color='#756bb1', lw=1.8, ms=8, label='Periodic')
ax.plot(WINDOWS, [best_mean[w]['min_av']*100 for w in WINDOWS],
        '^-', color='#756bb1', lw=2.6, ms=10, label='Aperiodic (avg-opt, best mean)')
ax.axvline(fwhm/2, color='orange', ls='--', lw=1.5,
           label=f'Bragg half-width {fwhm/2:.1f}°')
style_axes(ax, 'Window ±Δθ (°)', 'R_avg_min over ±Window (%)',
           'Worst-case (Min) R_avg over ±Window')
plt.tight_layout()
plt.savefig(ROOT / 'fig_avg_Rmin.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig_avg_Rmin.png")

# ── Fig 5 — per-layer profiles (best_mean) ──────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
cmap = plt.cm.plasma
N = None
for i, w in enumerate(WINDOWS):
    bl = best_mean[w]['bl']
    N = len(bl)
    idx = np.arange(1, N+1)
    d_pair = np.array([b[0] for b in bl])
    gamma  = np.array([b[1] for b in bl])
    d_Mo   = d_pair * gamma
    d_Si   = d_pair * (1 - gamma)
    col = cmap(i/(len(WINDOWS)-1))
    lab = f'±{w}°'
    axes[0][0].plot(idx, d_pair, 'o-', color=col, lw=1.5, ms=4, label=lab)
    axes[0][1].plot(idx, gamma,  's-', color=col, lw=1.5, ms=4, label=lab)
    axes[1][0].plot(idx, d_Mo,   '^-', color=col, lw=1.5, ms=4, label=lab)
    axes[1][1].plot(idx, d_Si,   'v-', color=col, lw=1.5, ms=4, label=lab)

axes[0][0].axhline(d0_per, color='k', ls=':', lw=1.5, label=f'Periodic {d0_per:.2f} nm')
axes[0][1].axhline(g0_per, color='k', ls=':', lw=1.5, label=f'Periodic {g0_per:.3f}')
axes[1][0].axhline(d0_per*g0_per, color='k', ls=':', lw=1.5,
                   label=f'Periodic {d0_per*g0_per:.2f} nm')
axes[1][1].axhline(d0_per*(1-g0_per), color='k', ls=':', lw=1.5,
                   label=f'Periodic {d0_per*(1-g0_per):.2f} nm')

titles = [
    'Bilayer thickness d_pair (nm)\nbounds [4, 14] nm',
    'Mo fraction γ = d_Mo / d_pair\nbounds [0.15, 0.70]',
    'Mo layer thickness d_Mo (nm)',
    'Si layer thickness d_Si (nm)',
]
ylabels = ['d_pair (nm)', 'γ', 'd_Mo (nm)', 'd_Si (nm)']
for ax, t, yl in zip(axes.flatten(), titles, ylabels):
    ax.set_xlabel(f'Bilayer index  (1 = top / vacuum-side,  {N} = substrate-side)',
                  fontsize=11)
    ax.set_ylabel(yl, fontsize=12)
    ax.set_title(t, fontsize=11)
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(fontsize=8, ncol=2, loc='best')
    ax.grid(alpha=0.3)
    ax.set_xlim(0.5, N+0.5)

plt.suptitle(f'Per-Layer Profiles — Best Aperiodic Designs by max R_avg_mean\n'
             f'(Unpolarized / avg-opt, Mo/Si N=40, λ=13.5 nm, θ₀=6°)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(ROOT / 'fig_avg_layers.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig_avg_layers.png")

print("\n══ SUMMARY (best by max R_avg_mean, unpolarized) ══")
print(f"{'window':>8} {'λ*':>5} {'k*':>3}  {'periodic':>26}  {'aperiodic':>26}")
print(f"{'':>8} {'':>5} {'':>3}  {'mean / peak / min':>26}  {'mean / peak / min':>26}")
for w in WINDOWS:
    bm = best_mean[w]
    pe = per[w]
    print(f"  ±{w:>4}° {bm['lam']:>4.1f} {bm['k']:>2}  "
          f"{pe['mean']*100:>6.2f} / {pe['peak']*100:>5.2f} / {pe['min']*100:>5.2f}    "
          f"{bm['mean_av']*100:>6.2f} / {R_avg(bm['bl'], THETA0_DEG)*100:>5.2f} / {bm['min_av']*100:>5.2f}")
