"""
Regenerate the two sub-panels (Mean Reflectivity vs window; R(θ) overlay)
from fig_best_mean.png as standalone figures, WITHOUT the λ annotations.
Criterion: max R_mean.
"""

import pickle, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path('/home/mingxuan/project/Litho')

# ── Load previous TMM results + reopen solver for periodic ref curve ──
LAMBDA_NM = 13.5
THETA0_DEG = 6.0
N_MO  = 0.921 + 1j * 0.0064
N_SI  = 0.999 + 1j * 0.0018
N_VAC = 1.0 + 0j

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

def sweep_theta(bl, th):
    return np.array([R_stack(bl, t) for t in th])

with open(ROOT / 'euv_best_results.pkl', 'rb') as f:
    data = pickle.load(f)

all_cands = data['all_cands']
per_evals = data['per_evals']
fwhm      = data['fwhm']
d0_per    = data['d0']
g0_per    = data['g0']
bl_per    = [(d0_per, g0_per)] * 40

WINDOWS = sorted({w for (w, _, _) in all_cands.keys()})
LAMBDAS = sorted({l for (_, l, _) in all_cands.keys()})
K       = 1 + max(k for (_, _, k) in all_cands.keys())

def pick_best(metric):
    best = {}
    for w in WINDOWS:
        cands = [all_cands[(w, l, k)] for l in LAMBDAS for k in range(K)]
        best[w] = max(cands, key=lambda e: e[metric])
    return best

best_mean = pick_best('R_mean')

per_mean = np.array([per_evals[w]['R_mean']*100 for w in WINDOWS])
ap_mean  = np.array([best_mean[w]['R_mean']*100 for w in WINDOWS])

# ══════ Figure 1 — R_mean vs window (clean) ══════
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(WINDOWS, per_mean, 's:', color='#1a9641', lw=1.8, ms=8,
        label='Periodic mean')
ax.plot(WINDOWS, ap_mean,  's-', color='#1a9641', lw=2.6, ms=10,
        label='Aperiodic mean (max R_mean)')
ax.axvline(fwhm/2, color='orange', ls='--', lw=1.4,
           label=f'Bragg half-width {fwhm/2:.1f}°')
ax.set_xlabel('Window ±Δθ (°)', fontsize=11)
ax.set_ylabel('R_mean (%)', fontsize=11)
ax.set_title('Mean Reflectivity over ±Window', fontsize=12)
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(ROOT / 'fig_best_mean_clean_Rmean.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig_best_mean_clean_Rmean.png")

# ══════ Figure 2 — R(θ) overlay (clean, larger fonts) ══════
fig, ax = plt.subplots(figsize=(11, 7.5))
cmap = plt.cm.plasma
for i, w in enumerate(WINDOWS):
    e = best_mean[w]
    col = cmap(i/(len(WINDOWS)-1))
    ax.plot(e['thetas'], e['R_arr']*100, '-', lw=2.2, color=col,
            label=f'±{w}°')
th_ref = np.linspace(THETA0_DEG-15, THETA0_DEG+15, 240)
ax.plot(th_ref, sweep_theta(bl_per, th_ref)*100, 'k:', lw=2.2, alpha=0.85,
        label='Periodic (ref)')
ax.axvline(THETA0_DEG, color='gray', ls=':', lw=1)
ax.set_xlabel('θ (°)', fontsize=15)
ax.set_ylabel('R (%)', fontsize=15)
ax.set_title('R(θ) of best designs per window  (criterion: max R_mean)', fontsize=15)
ax.tick_params(axis='both', labelsize=12)
ax.legend(fontsize=12, ncol=2, loc='lower center')
ax.grid(alpha=0.3); ax.set_ylim(0, None)
plt.tight_layout()
plt.savefig(ROOT / 'fig_best_mean_clean_Rtheta.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig_best_mean_clean_Rtheta.png")
