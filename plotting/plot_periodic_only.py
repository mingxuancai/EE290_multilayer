"""
Standalone plot: only the periodic (avg) R(θ) curve from the window-sweep figure.
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
bl_per_avg = data['bl_per_avg']
d0 = bl_per_avg[0][0]
g0 = bl_per_avg[0][1]

th = np.linspace(THETA0_DEG-15, THETA0_DEG+15, 401)
R = np.array([R_avg(bl_per_avg, t) for t in th]) * 100

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.plot(th, R, 'k-', lw=2.5, label=f'Periodic (avg)  d={d0:.2f} nm, γ={g0:.3f}')
ax.axvline(THETA0_DEG, color='gray', ls=':', lw=1)
ax.set_xlabel('θ (°)', fontsize=14)
ax.set_ylabel('R_avg (%)', fontsize=14)
ax.set_title('Periodic Multilayer R_avg(θ)  (Mo/Si N=40, λ=13.5 nm)', fontsize=14)
ax.tick_params(axis='both', labelsize=12)
ax.legend(fontsize=12); ax.grid(alpha=0.3)
ax.set_ylim(0, None)
plt.tight_layout()
plt.savefig(ROOT / 'fig_periodic_avg_only.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig_periodic_avg_only.png")
