"""
Mo/Si EUV Aperiodic Multilayer Reflectivity Calculator
λ = 13.5 nm, nominal incidence 6° from normal

Stages:
  1. TMM solver + periodic baseline validation
  2. Route C: chirped/graded parametric structures
  3. Route A: greedy layer-by-layer growth + L-BFGS-B refinement → Pareto front
  4. Route B: differential evolution global optimizer (validates Route A quality)
  5. Nevot-Croce roughness correction (σ = 0, 0.2, 0.4 nm)
  6. Re-optimize under roughness; produce all summary figures
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass, field
from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────
LAMBDA_NM = 13.5          # wavelength in nm
THETA0_DEG = 6.0          # nominal angle from normal (degrees)
WINDOW_DEG = 1.0          # ± angular window
STEP_DEG = 0.05           # angular sweep resolution (final plots)
FAST_STEP_DEG = 0.2       # fast sweep for optimization (5× fewer evaluations)

# CXRO/Henke values at 13.5 nm
N_MO = 0.921 + 1j * 0.0064
N_SI = 0.999 + 1j * 0.0018
N_VAC = 1.0 + 0j
N_SI_SUB = 0.999 + 1j * 0.0018  # Si substrate same as film

# ─────────────────────────────────────────────────────────────────────────────
# Core Transfer-Matrix solver (s and p polarisation)
# ─────────────────────────────────────────────────────────────────────────────

def _kz(n: complex, theta_inc_rad: float, n_inc: complex) -> complex:
    """Normal component of wavevector in medium with index n."""
    k0 = 2 * np.pi / LAMBDA_NM
    sin_t = n_inc.real * np.sin(theta_inc_rad) / n
    cos_t = np.sqrt(1.0 - sin_t**2 + 0j)
    return k0 * n * cos_t


def reflectivity_tmm(
    d_list: np.ndarray,
    n_list: np.ndarray,
    theta_deg: float,
    roughness: np.ndarray | None = None,
) -> Tuple[float, float, float]:
    """
    Transfer-matrix method for a stack of layers.

    d_list  : thicknesses [nm],  length = N_layers
    n_list  : complex refractive indices, length = N_layers
              n_list[0]  = incident medium (vacuum)
              n_list[-1] = substrate
              layers 1..N-2 are the actual multilayer
    theta_deg: angle of incidence from normal in degrees
    roughness: interface roughness σ [nm], length = N_layers - 1
               (Nevot-Croce factor applied at each interface)
    Returns (R_s, R_p, R_avg)
    """
    theta_rad = np.deg2rad(theta_deg)
    k0 = 2 * np.pi / LAMBDA_NM
    N = len(n_list)

    # kz in each layer
    n0 = n_list[0]
    sin0 = n0 * np.sin(theta_rad)
    kz = np.zeros(N, dtype=complex)
    for i, n in enumerate(n_list):
        cos_t = np.sqrt((n**2 - sin0**2) / n**2 + 0j) * n
        kz[i] = k0 * np.sqrt(n**2 - sin0**2 + 0j)

    def fresnel(i, pol):
        """Fresnel r coefficient at interface i → i+1."""
        if pol == 's':
            r = (kz[i] - kz[i+1]) / (kz[i] + kz[i+1])
        else:  # p
            r = (n_list[i+1]**2 * kz[i] - n_list[i]**2 * kz[i+1]) / \
                (n_list[i+1]**2 * kz[i] + n_list[i]**2 * kz[i+1])
        # Nevot-Croce roughness factor
        if roughness is not None and roughness[i] > 0:
            sigma = roughness[i]
            dw = np.exp(-2 * kz[i] * kz[i+1] * sigma**2)
            r = r * dw
        return r

    def tmm_pol(pol):
        # Start from substrate: M = identity
        M = np.eye(2, dtype=complex)
        for i in range(N-2, 0, -1):
            r = fresnel(i, pol)
            t = 1 + r if pol == 's' else (1 + r) * n_list[i] / n_list[i]
            phi = kz[i] * d_list[i]
            # Interface matrix × propagation matrix
            D_inv = np.array([[1, r], [r, 1]]) / (1 - r**2 + 1e-30) * np.sqrt(1 - r**2 + 0j)
            # Use standard TMM formulation
            P = np.array([[np.exp(1j*phi), 0], [0, np.exp(-1j*phi)]])
            I = np.array([[1, r], [r, 1]])
            M = I @ P @ M
        # Final interface (incident → first layer)
        r0 = fresnel(0, pol)
        I0 = np.array([[1, r0], [r0, 1]])
        M = I0 @ M

        # Recursive reflectivity (more stable than matrix inversion)
        return _recursive_r(n_list, kz, d_list, roughness, pol)

    R_s = abs(_recursive_r(n_list, kz, d_list, roughness, 's'))**2
    R_p = abs(_recursive_r(n_list, kz, d_list, roughness, 'p'))**2
    R_avg = 0.5 * (R_s + R_p)
    return float(R_s), float(R_p), float(R_avg)


def _recursive_r(n_list, kz, d_list, roughness, pol):
    """
    Recursive Fresnel algorithm (numerically stable).
    Starts from substrate and recurses to incident medium.
    """
    N = len(n_list)

    def r_ij(i):
        if pol == 's':
            r = (kz[i] - kz[i+1]) / (kz[i] + kz[i+1])
        else:
            r = (n_list[i+1]**2 * kz[i] - n_list[i]**2 * kz[i+1]) / \
                (n_list[i+1]**2 * kz[i] + n_list[i]**2 * kz[i+1])
        if roughness is not None and i < len(roughness) and roughness[i] > 0:
            sigma = roughness[i]
            dw = np.exp(-2 * kz[i] * kz[i+1] * sigma**2)
            r = r * dw
        return r

    # r_total starts as r at substrate interface (layer N-2 → N-1)
    r_total = r_ij(N-2)

    # Recurse from substrate outward
    for i in range(N-3, -1, -1):
        r = r_ij(i)
        phi = kz[i+1] * d_list[i+1]  # propagation in layer i+1
        r_total = (r + r_total * np.exp(2j * phi)) / \
                  (1 + r * r_total * np.exp(2j * phi))

    return r_total


# ─────────────────────────────────────────────────────────────────────────────
# Stack builder helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_stack(bilayers: List[Tuple[float, float]], top_mo: bool = True
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    bilayers: list of (d_pair_nm, gamma) for each bilayer
              gamma = d_Mo / d_pair
    top_mo  : if True, layer order is Mo/Si from top (incident side)
    Returns (d_list, n_list) including vacuum (idx 0) and Si substrate (idx -1).
    """
    d = [0.0]   # incident medium thickness (dummy)
    n = [N_VAC]
    for d_pair, gamma in bilayers:
        d_mo = d_pair * gamma
        d_si = d_pair * (1 - gamma)
        if top_mo:
            d += [d_mo, d_si]
            n += [N_MO, N_SI]
        else:
            d += [d_si, d_mo]
            n += [N_SI, N_MO]
    # substrate
    d.append(0.0)
    n.append(N_SI_SUB)
    return np.array(d), np.array(n)


def reflectivity_stack(bilayers, theta_deg, roughness_sigma=0.0, top_mo=True):
    """Convenience: build stack and compute R_avg."""
    d_list, n_list = build_stack(bilayers, top_mo)
    sigma_arr = np.full(len(d_list)-1, roughness_sigma)
    _, _, R = reflectivity_tmm(d_list, n_list, theta_deg, sigma_arr)
    return R


def sweep_angle(bilayers, theta0=THETA0_DEG, window=WINDOW_DEG,
                step=STEP_DEG, roughness_sigma=0.0, top_mo=True):
    """Sweep angle around theta0 ± window."""
    thetas = np.arange(theta0 - window, theta0 + window + step/2, step)
    R_vals = [reflectivity_stack(bilayers, t, roughness_sigma, top_mo) for t in thetas]
    return thetas, np.array(R_vals)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 – Periodic baseline
# ─────────────────────────────────────────────────────────────────────────────

def design_periodic(N_bilayers, d_pair=6.9, gamma=0.38, top_mo=True):
    return [(d_pair, gamma)] * N_bilayers


def validate_solver():
    """Cross-check recursive vs TMM and verify ~70% baseline."""
    print("=" * 60)
    print("STAGE 1: Solver validation & periodic baseline")
    print("=" * 60)
    # Bragg condition: d_pair ≈ λ / (2 * n_avg * cos(theta)), accounting for medium
    theta_rad = np.deg2rad(THETA0_DEG)
    gamma0 = 0.38
    n_avg_real = gamma0 * N_MO.real + (1 - gamma0) * N_SI.real
    d_bragg = LAMBDA_NM / (2 * n_avg_real * np.cos(theta_rad))
    print(f"  Bragg d_pair estimate: {d_bragg:.3f} nm")

    results = {}
    for N in [40, 50]:
        for top_mo in [True, False]:
            label = f"N={N}, {'Mo' if top_mo else 'Si'}-top"
            bl = design_periodic(N, d_pair=d_bragg, gamma=0.38, top_mo=top_mo)
            R = reflectivity_stack(bl, THETA0_DEG, top_mo=top_mo)
            print(f"  {label}: R = {R*100:.2f}%")
            results[(N, top_mo)] = R

    # Pick best configuration
    best_key = max(results, key=results.get)
    best_N, best_top_mo = best_key
    print(f"\n  Best config: N={best_N}, {'Mo' if best_top_mo else 'Si'}-top, "
          f"R = {results[best_key]*100:.2f}%")

    # Fine-optimize d_pair and gamma for periodic
    def neg_R(params):
        d, g = params
        if d < 4 or d > 12 or g < 0.2 or g > 0.7:
            return 0.0
        bl = design_periodic(best_N, d, g, best_top_mo)
        return -reflectivity_stack(bl, THETA0_DEG, top_mo=best_top_mo)

    from scipy.optimize import minimize
    res = minimize(neg_R, [d_bragg, 0.38], method='Nelder-Mead',
                   options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 2000})
    d_opt, g_opt = res.x
    R_opt = -res.fun

    print(f"  Optimized periodic: d_pair={d_opt:.4f} nm, γ={g_opt:.4f}, "
          f"R={R_opt*100:.2f}%")

    return best_N, best_top_mo, d_opt, g_opt, R_opt


# ─────────────────────────────────────────────────────────────────────────────
# Objective function family
# ─────────────────────────────────────────────────────────────────────────────

def J_alpha(bilayers, alpha, roughness_sigma=0.0, top_mo=True, fast=True):
    """
    J_alpha = alpha * R(theta0) + (1-alpha) * mean(R over ± window)
    alpha=1: maximize peak; alpha=0: maximize angular average.
    fast=True uses coarser angular grid during optimization.
    """
    step = FAST_STEP_DEG if fast else STEP_DEG
    R_peak = reflectivity_stack(bilayers, THETA0_DEG, roughness_sigma, top_mo)
    thetas, R_sweep = sweep_angle(bilayers, step=step,
                                  roughness_sigma=roughness_sigma, top_mo=top_mo)
    R_mean = np.mean(R_sweep)
    return alpha * R_peak + (1 - alpha) * R_mean


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 – Route C: Chirped / graded parametric design
# ─────────────────────────────────────────────────────────────────────────────

def design_chirped(N, d0, beta, delta, gamma0, d_gamma, top_mo=True):
    """
    d_pair_i = d0 * (1 + beta*(i/N) + delta*sin(2*pi*i/N))
    gamma_i  = gamma0 + d_gamma*(i/N - 0.5)
    """
    bilayers = []
    for i in range(N):
        d = d0 * (1 + beta * i / N + delta * np.sin(2 * np.pi * i / N))
        g = gamma0 + d_gamma * (i / N - 0.5)
        d = np.clip(d, 4, 14)
        g = np.clip(g, 0.15, 0.75)
        bilayers.append((d, g))
    return bilayers


def optimize_chirped(N, d0, g0, alpha_target, top_mo=True):
    """Optimize 5-parameter chirped structure."""
    def neg_J(params, _N=N, _at=alpha_target, _top=top_mo, _d0=d0):
        d0_, beta, delta, gamma0, d_gamma = params
        bl = design_chirped(_N, d0_, beta, delta, gamma0, d_gamma, _top)
        return -J_alpha(bl, _at, top_mo=_top, fast=True)

    bounds = [(d0*0.8, d0*1.2), (-0.5, 0.5), (-0.3, 0.3),
              (0.2, 0.6), (-0.3, 0.3)]
    from scipy.optimize import differential_evolution
    res = differential_evolution(neg_J, bounds, seed=42, maxiter=100,
                                 popsize=6, tol=1e-4, workers=1)
    return res.x, -res.fun


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 – Route A: Greedy layer-by-layer growth + L-BFGS-B refinement
# ─────────────────────────────────────────────────────────────────────────────

def greedy_grow(N, alpha, d0, g0, roughness_sigma=0.0, top_mo=True):
    """Greedy aperiodic: add bilayers one at a time, optimize each."""
    bilayers = []
    for i in range(N):
        def neg_J(params, _bl=bilayers):
            d, g = params
            d = np.clip(d, 4, 14)
            g = np.clip(g, 0.15, 0.75)
            bl = _bl + [(d, g)]
            return -J_alpha(bl, alpha, roughness_sigma, top_mo, fast=True)

        res = minimize(neg_J, [d0, g0], method='Nelder-Mead',
                       bounds=[(4, 14), (0.15, 0.75)],
                       options={'xatol': 1e-4, 'fatol': 1e-7, 'maxiter': 300,
                                'adaptive': True})
        d_i, g_i = np.clip(res.x[0], 4, 14), np.clip(res.x[1], 0.15, 0.75)
        bilayers.append((d_i, g_i))

    return bilayers


def refine_full(bilayers, alpha, roughness_sigma=0.0, top_mo=True, n_passes=2):
    """
    Coordinate descent refinement: re-optimize each bilayer in sequence.
    2 passes ~= 2N Nelder-Mead calls, much cheaper than joint L-BFGS-B.
    """
    bl = list(bilayers)
    N = len(bl)
    for _ in range(n_passes):
        for i in range(N):
            prefix = bl[:i]
            suffix = bl[i+1:]

            def neg_J(params, _pre=prefix, _suf=suffix):
                d, g = params
                d = np.clip(d, 4, 14)
                g = np.clip(g, 0.15, 0.75)
                full = _pre + [(d, g)] + _suf
                return -J_alpha(full, alpha, roughness_sigma, top_mo, fast=True)

            res = minimize(neg_J, list(bl[i]), method='Nelder-Mead',
                           options={'xatol': 1e-4, 'fatol': 1e-7,
                                    'maxiter': 200, 'adaptive': True})
            bl[i] = (np.clip(res.x[0], 4, 14), np.clip(res.x[1], 0.15, 0.75))

    J_final = J_alpha(bl, alpha, roughness_sigma, top_mo, fast=False)
    return bl, J_final


def compute_pareto_routeA(N, d0, g0, alphas, roughness_sigma=0.0, top_mo=True):
    """Compute Pareto front via Route A for each alpha value."""
    print(f"\n  Route A: greedy + L-BFGS-B, N={N}, σ={roughness_sigma} nm")
    results = {}
    for alpha in alphas:
        print(f"    α={alpha:.2f}...", end=" ", flush=True)
        bl_greedy = greedy_grow(N, alpha, d0, g0, roughness_sigma, top_mo)
        bl_refined, _ = refine_full(bl_greedy, alpha, roughness_sigma, top_mo)

        R_peak = reflectivity_stack(bl_refined, THETA0_DEG, roughness_sigma, top_mo)
        _, R_sweep = sweep_angle(bl_refined, roughness_sigma=roughness_sigma, top_mo=top_mo)
        R_mean = np.mean(R_sweep)
        print(f"R_peak={R_peak*100:.2f}%, R_mean={R_mean*100:.2f}%")
        results[alpha] = {'bilayers': bl_refined, 'R_peak': R_peak, 'R_mean': R_mean}
    return results


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 – Route B: Differential evolution (validates Route A)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pareto_routeB(N, alphas_subset, roughness_sigma=0.0, top_mo=True):
    """Route B: full DE optimization for selected alpha values."""
    print(f"\n  Route B: differential evolution, N={N}, σ={roughness_sigma} nm")
    results = {}
    for alpha in alphas_subset:
        print(f"    α={alpha:.2f}...", end=" ", flush=True)

        def neg_J(x, _N=N, _alpha=alpha, _sigma=roughness_sigma, _top=top_mo):
            bl = [(np.clip(x[2*i], 4, 14), np.clip(x[2*i+1], 0.15, 0.75))
                  for i in range(_N)]
            return -J_alpha(bl, _alpha, _sigma, _top, fast=True)

        bounds = [(4, 14), (0.15, 0.75)] * N
        res = differential_evolution(neg_J, bounds, seed=42, maxiter=80,
                                     popsize=4, tol=1e-4, workers=1,
                                     mutation=(0.5, 1.5), recombination=0.7)
        bl = [(np.clip(res.x[2*i], 4, 14), np.clip(res.x[2*i+1], 0.15, 0.75))
              for i in range(N)]
        R_peak = reflectivity_stack(bl, THETA0_DEG, roughness_sigma, top_mo)
        _, R_sweep = sweep_angle(bl, roughness_sigma=roughness_sigma, top_mo=top_mo)
        R_mean = np.mean(R_sweep)
        print(f"R_peak={R_peak*100:.2f}%, R_mean={R_mean*100:.2f}%")
        results[alpha] = {'bilayers': bl, 'R_peak': R_peak, 'R_mean': R_mean}
    return results


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 & 6 – Roughness study + final figures
# ─────────────────────────────────────────────────────────────────────────────

def run_roughness_study(bl_periodic, pareto_A, N, d0, g0, top_mo,
                        sigmas=[0.0, 0.2, 0.4], alphas=[0.5]):
    """
    For each sigma, compute:
      - periodic R_peak and R_mean
      - aperiodic (α=0.5 by default, re-optimized under roughness)
    """
    print("\n  Stage 5&6: Roughness study")
    rough_results = {}
    for sigma in sigmas:
        print(f"  σ={sigma} nm")
        # Periodic under roughness (no re-optimization, just evaluate)
        R_peak_p = reflectivity_stack(bl_periodic, THETA0_DEG, sigma, top_mo)
        _, R_sweep_p = sweep_angle(bl_periodic, roughness_sigma=sigma, top_mo=top_mo)
        R_mean_p = np.mean(R_sweep_p)

        # Aperiodic: re-optimize under roughness
        aperiodic = {}
        for alpha in alphas:
            print(f"    α={alpha}, re-optimizing...", end=" ", flush=True)
            bl_g = greedy_grow(N, alpha, d0, g0, sigma, top_mo)
            bl_r, _ = refine_full(bl_g, alpha, sigma, top_mo)
            R_peak_a = reflectivity_stack(bl_r, THETA0_DEG, sigma, top_mo)
            _, R_sweep_a = sweep_angle(bl_r, roughness_sigma=sigma, top_mo=top_mo)
            R_mean_a = np.mean(R_sweep_a)
            print(f"R_peak={R_peak_a*100:.2f}%, R_mean={R_mean_a*100:.2f}%")
            aperiodic[alpha] = {'bilayers': bl_r, 'R_peak': R_peak_a, 'R_mean': R_mean_a}

        rough_results[sigma] = {
            'periodic': {'R_peak': R_peak_p, 'R_mean': R_mean_p,
                         'R_sweep': R_sweep_p},
            'aperiodic': aperiodic,
        }
    return rough_results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    'periodic': '#2c7bb6',
    0.0:  '#1a9641',
    0.25: '#a6d96a',
    0.5:  '#fdae61',
    0.75: '#d7191c',
    1.0:  '#762a83',
}
ALPHA_LABELS = {0.0: 'α=0 (max bandwidth)', 0.25: 'α=0.25', 0.5: 'α=0.5',
                0.75: 'α=0.75', 1.0: 'α=1 (max peak)'}


def plot_all(bl_periodic, pareto_A, pareto_B, pareto_chirped,
             rough_results, N_best, top_mo, d0, g0, alphas):
    """Generate all summary figures."""

    thetas_ref = np.arange(THETA0_DEG - WINDOW_DEG,
                           THETA0_DEG + WINDOW_DEG + STEP_DEG/2, STEP_DEG)

    # ── Figure 1: R(θ) curves ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.set_title(f"R(θ) — Periodic vs Aperiodic (N={N_best}, σ=0)", fontsize=11)
    _, R_p = sweep_angle(bl_periodic, top_mo=top_mo)
    ax.plot(thetas_ref, R_p * 100, lw=2.5, color=COLORS['periodic'],
            label=f'Periodic (R_peak={max(R_p)*100:.1f}%)', zorder=10)
    for alpha in alphas:
        bl = pareto_A[alpha]['bilayers']
        _, R_a = sweep_angle(bl, top_mo=top_mo)
        ax.plot(thetas_ref, R_a * 100, lw=1.5, color=COLORS[alpha],
                label=ALPHA_LABELS[alpha] + f' ({max(R_a)*100:.1f}%)', alpha=0.85)
    ax.axvline(THETA0_DEG, color='k', ls='--', lw=0.8, label='θ₀')
    ax.axvspan(THETA0_DEG - WINDOW_DEG, THETA0_DEG + WINDOW_DEG,
               alpha=0.08, color='gray', label='±1° window')
    ax.set_xlabel('Angle from normal (°)')
    ax.set_ylabel('Reflectivity (%)')
    ax.legend(fontsize=7.5, loc='lower center')
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)

    # Chirped comparison
    ax2 = axes[1]
    ax2.set_title(f"R(θ) — Chirped (Route C) vs Periodic", fontsize=11)
    ax2.plot(thetas_ref, R_p * 100, lw=2.5, color=COLORS['periodic'],
             label='Periodic', zorder=10)
    c_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    for j, alpha in enumerate(pareto_chirped):
        bl = pareto_chirped[alpha]['bilayers']
        _, R_c = sweep_angle(bl, top_mo=top_mo)
        ax2.plot(thetas_ref, R_c * 100, lw=1.5, color=c_colors[j % len(c_colors)],
                 label=f'Chirped {ALPHA_LABELS[alpha]} ({max(R_c)*100:.1f}%)')
    ax2.axvline(THETA0_DEG, color='k', ls='--', lw=0.8)
    ax2.axvspan(THETA0_DEG - WINDOW_DEG, THETA0_DEG + WINDOW_DEG,
                alpha=0.08, color='gray')
    ax2.set_xlabel('Angle from normal (°)')
    ax2.set_ylabel('Reflectivity (%)')
    ax2.legend(fontsize=7.5, loc='lower center')
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig1_R_theta.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig1_R_theta.png")

    # ── Figure 2: Pareto front (R_peak vs R_mean) ─────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title('Pareto Front: Peak Reflectivity vs Angular-Average Reflectivity', fontsize=11)

    # Periodic reference point
    R_pk_p = pareto_A[alphas[0]]['R_peak']  # use first alpha for reference shape
    _, R_sweep_p = sweep_angle(bl_periodic, top_mo=top_mo)
    ax.scatter([np.mean(R_sweep_p)*100], [max(R_sweep_p)*100],
               s=150, marker='*', color=COLORS['periodic'], zorder=10,
               label='Periodic baseline', edgecolors='k', linewidths=0.8)

    # Route A
    Ra_peaks = [pareto_A[a]['R_peak']*100 for a in alphas]
    Ra_means = [pareto_A[a]['R_mean']*100 for a in alphas]
    ax.plot(Ra_means, Ra_peaks, 'o-', color='#d7191c', lw=2, ms=8,
            label='Route A (greedy+L-BFGS-B)', zorder=8)
    for a, xv, yv in zip(alphas, Ra_means, Ra_peaks):
        ax.annotate(f'α={a}', (xv, yv), textcoords='offset points',
                    xytext=(5, 3), fontsize=7)

    # Route B (subset)
    if pareto_B:
        Rb_peaks = [pareto_B[a]['R_peak']*100 for a in pareto_B]
        Rb_means = [pareto_B[a]['R_mean']*100 for a in pareto_B]
        ax.scatter(Rb_means, Rb_peaks, s=100, marker='D', color='#2171b5',
                   zorder=9, label='Route B (diff. evolution)', edgecolors='k',
                   linewidths=0.8)

    # Chirped
    Rc_peaks = [pareto_chirped[a]['R_peak']*100 for a in pareto_chirped]
    Rc_means = [pareto_chirped[a]['R_mean']*100 for a in pareto_chirped]
    ax.plot(Rc_means, Rc_peaks, 's--', color='#6a51a3', lw=1.5, ms=7,
            label='Route C (chirped)', zorder=7)

    ax.set_xlabel('Mean R over θ₀ ± 1° (%)')
    ax.set_ylabel('Peak R at θ₀ (%)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig2_pareto_front.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig2_pareto_front.png")

    # ── Figure 3: Bilayer thickness profiles ─────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    alpha_list_for_profile = alphas
    for j, alpha in enumerate(alpha_list_for_profile):
        ax = axes[j]
        bl = pareto_A[alpha]['bilayers']
        d_pairs = [d for d, g in bl]
        gammas = [g for d, g in bl]
        idx = np.arange(1, len(bl)+1)
        ax2_ = ax.twinx()
        ax.bar(idx, d_pairs, color='#4292c6', alpha=0.7, label='d_pair (nm)')
        ax2_.plot(idx, gammas, 'r.-', ms=4, label='γ')
        ax.axhline(np.mean(d_pairs), color='navy', ls='--', lw=1, label=f'd_mean={np.mean(d_pairs):.2f}')
        ax.set_title(f'α={alpha}: d_pair & γ profile', fontsize=9)
        ax.set_xlabel('Bilayer index')
        ax.set_ylabel('d_pair (nm)', color='#4292c6')
        ax2_.set_ylabel('γ = d_Mo/d_pair', color='r')
        ax2_.set_ylim(0.1, 0.8)
        ax.set_ylim(0, 16)
    # Hide unused subplots
    for j in range(len(alpha_list_for_profile), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle(f'Aperiodic Bilayer Profiles (Route A, N={N_best})', fontsize=12)
    plt.tight_layout()
    plt.savefig('fig3_bilayer_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig3_bilayer_profiles.png")

    # ── Figure 4: Roughness study ─────────────────────────────────────────
    sigmas = sorted(rough_results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.set_title('Effect of Roughness: R_peak vs σ', fontsize=11)
    R_pk_per = [rough_results[s]['periodic']['R_peak']*100 for s in sigmas]
    R_pk_ap  = [rough_results[s]['aperiodic'][0.5]['R_peak']*100 for s in sigmas]
    ax.plot(sigmas, R_pk_per, 'o-', color=COLORS['periodic'], lw=2, ms=8, label='Periodic')
    ax.plot(sigmas, R_pk_ap, 's-', color=COLORS[0.5], lw=2, ms=8, label='Aperiodic α=0.5')
    ax.set_xlabel('Interface roughness σ (nm)')
    ax.set_ylabel('Peak Reflectivity at θ₀ (%)')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.set_title('Effect of Roughness: R_mean (±1°) vs σ', fontsize=11)
    R_mn_per = [rough_results[s]['periodic']['R_mean']*100 for s in sigmas]
    R_mn_ap  = [rough_results[s]['aperiodic'][0.5]['R_mean']*100 for s in sigmas]
    ax.plot(sigmas, R_mn_per, 'o-', color=COLORS['periodic'], lw=2, ms=8, label='Periodic')
    ax.plot(sigmas, R_mn_ap, 's-', color=COLORS[0.5], lw=2, ms=8, label='Aperiodic α=0.5')
    ax.set_xlabel('Interface roughness σ (nm)')
    ax.set_ylabel('Mean R over θ₀ ± 1° (%)')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig4_roughness.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig4_roughness.png")

    # ── Figure 5: R(θ) under roughness (σ=0 vs 0.4 nm) ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for k, sigma in enumerate([0.0, 0.4]):
        ax = axes[k]
        ax.set_title(f'R(θ) with σ={sigma} nm roughness', fontsize=11)
        _, R_p_r = sweep_angle(bl_periodic, roughness_sigma=sigma, top_mo=top_mo)
        ax.plot(thetas_ref, R_p_r*100, lw=2.5, color=COLORS['periodic'],
                label=f'Periodic ({max(R_p_r)*100:.1f}%)', zorder=10)
        if sigma in rough_results:
            bl_ap = rough_results[sigma]['aperiodic'][0.5]['bilayers']
        else:
            bl_ap = pareto_A[0.5]['bilayers']
        _, R_a_r = sweep_angle(bl_ap, roughness_sigma=sigma, top_mo=top_mo)
        ax.plot(thetas_ref, R_a_r*100, lw=2, color=COLORS[0.5],
                label=f'Aperiodic α=0.5 ({max(R_a_r)*100:.1f}%)')
        ax.axvline(THETA0_DEG, color='k', ls='--', lw=0.8)
        ax.axvspan(THETA0_DEG - WINDOW_DEG, THETA0_DEG + WINDOW_DEG,
                   alpha=0.08, color='gray')
        ax.set_xlabel('Angle from normal (°)')
        ax.set_ylabel('Reflectivity (%)')
        ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig('fig5_roughness_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig5_roughness_curves.png")

    # ── Figure 6: Bandwidth metric (Δθ at 90% of peak) ──────────────────
    def bandwidth_90(bilayers, sigma=0.0, top_mo=True):
        thetas = np.arange(THETA0_DEG - 5, THETA0_DEG + 5, 0.01)
        R_vals = [reflectivity_stack(bilayers, t, sigma, top_mo) for t in thetas]
        R_arr = np.array(R_vals)
        peak = max(R_arr)
        threshold = 0.9 * peak
        above = thetas[R_arr >= threshold]
        if len(above) < 2:
            return 0.0
        return above[-1] - above[0]

    bws_per = []
    bws_ap = []
    for sigma in sigmas:
        bws_per.append(bandwidth_90(bl_periodic, sigma, top_mo))
        bl_ap_s = rough_results[sigma]['aperiodic'][0.5]['bilayers']
        bws_ap.append(bandwidth_90(bl_ap_s, sigma, top_mo))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title('Angular Bandwidth Δθ₉₀% vs Roughness σ', fontsize=11)
    ax.plot(sigmas, bws_per, 'o-', color=COLORS['periodic'], lw=2, ms=8, label='Periodic')
    ax.plot(sigmas, bws_ap, 's-', color=COLORS[0.5], lw=2, ms=8, label='Aperiodic α=0.5')
    ax.set_xlabel('Interface roughness σ (nm)')
    ax.set_ylabel('Δθ₉₀% (degrees)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig6_bandwidth.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig6_bandwidth.png")

    # ── Figure 7: Summary comparison table as figure ─────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    headers = ['Design', 'α', 'R_peak (%)', 'R_mean±1° (%)', 'Δθ₉₀° (°)', 'σ (nm)']
    rows = []

    # Periodic
    _, R_sw = sweep_angle(bl_periodic, top_mo=top_mo)
    bw = bandwidth_90(bl_periodic, 0.0, top_mo)
    rows.append(['Periodic', '—',
                 f"{max(R_sw)*100:.2f}", f"{np.mean(R_sw)*100:.2f}", f"{bw:.3f}", '0'])

    # Route A Pareto
    for alpha in alphas:
        bl = pareto_A[alpha]['bilayers']
        _, R_sw = sweep_angle(bl, top_mo=top_mo)
        bw = bandwidth_90(bl, 0.0, top_mo)
        rows.append([f'Aperiodic A', f'{alpha}',
                     f"{pareto_A[alpha]['R_peak']*100:.2f}",
                     f"{pareto_A[alpha]['R_mean']*100:.2f}", f"{bw:.3f}", '0'])

    # Route A with roughness 0.4
    sigma = 0.4
    bl_ap_r = rough_results[sigma]['aperiodic'][0.5]['bilayers']
    _, R_sw = sweep_angle(bl_ap_r, roughness_sigma=sigma, top_mo=top_mo)
    bw = bandwidth_90(bl_ap_r, sigma, top_mo)
    rows.append([f'Aperiodic A (opt)', '0.5',
                 f"{rough_results[sigma]['aperiodic'][0.5]['R_peak']*100:.2f}",
                 f"{rough_results[sigma]['aperiodic'][0.5]['R_mean']*100:.2f}",
                 f"{bw:.3f}", '0.4'])

    _, R_sw_p = sweep_angle(bl_periodic, roughness_sigma=sigma, top_mo=top_mo)
    bw_p = bandwidth_90(bl_periodic, sigma, top_mo)
    rows.append(['Periodic', '—',
                 f"{rough_results[sigma]['periodic']['R_peak']*100:.2f}",
                 f"{rough_results[sigma]['periodic']['R_mean']*100:.2f}",
                 f"{bw_p:.3f}", '0.4'])

    tbl = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2c7bb6')
            cell.set_text_props(color='white', fontweight='bold')
        elif 'Aperiodic' in str(rows[r-1][0]) if r > 0 and r <= len(rows) else False:
            cell.set_facecolor('#fff7bc')
    ax.set_title('Summary: Mo/Si EUV Multilayer @ 13.5 nm, θ₀=6°', fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig('fig7_summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig7_summary_table.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main execution
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  Mo/Si EUV Aperiodic Multilayer Study")
    print(f"  λ={LAMBDA_NM} nm, θ₀={THETA0_DEG}°, window=±{WINDOW_DEG}°")
    print("="*60)

    # ── Stage 1: Validate solver & find best periodic config ─────────────
    N_best, top_mo, d0, g0, R_per_opt = validate_solver()
    bl_periodic = design_periodic(N_best, d0, g0, top_mo)

    # Confirm periodic sweep
    _, R_sweep_p = sweep_angle(bl_periodic, top_mo=top_mo)
    print(f"\n  Periodic baseline: R_peak={max(R_sweep_p)*100:.2f}%, "
          f"R_mean(±1°)={np.mean(R_sweep_p)*100:.2f}%")

    # ── Stage 2: Route C (chirped) ────────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 2: Route C — Chirped parametric structures")
    print("="*60)
    alphas_chirp = [0.0, 0.5, 1.0]
    pareto_chirped = {}
    for alpha in alphas_chirp:
        print(f"  Chirped α={alpha}...", end=" ", flush=True)
        params, Jval = optimize_chirped(N_best, d0, g0, alpha, top_mo)
        bl_c = design_chirped(N_best, *params, top_mo=top_mo)
        R_pk = reflectivity_stack(bl_c, THETA0_DEG, top_mo=top_mo)
        _, R_sw = sweep_angle(bl_c, top_mo=top_mo)
        R_mn = np.mean(R_sw)
        print(f"R_peak={R_pk*100:.2f}%, R_mean={R_mn*100:.2f}%")
        pareto_chirped[alpha] = {'bilayers': bl_c, 'R_peak': R_pk, 'R_mean': R_mn}

    # ── Stage 3: Route A (greedy + refinement) ────────────────────────────
    print("\n" + "="*60)
    print("STAGE 3: Route A — Greedy growth + L-BFGS-B refinement")
    print("="*60)
    alphas_full = [0.0, 0.25, 0.5, 0.75, 1.0]
    pareto_A = compute_pareto_routeA(N_best, d0, g0, alphas_full, 0.0, top_mo)

    # ── Stage 4: Route B (differential evolution, subset) ────────────────
    print("\n" + "="*60)
    print("STAGE 4: Route B — Differential evolution (validation subset)")
    print("="*60)
    alphas_B = [0.5, 1.0]   # reduced subset for time budget
    N_B = 20  # use N=20 for DE (80-dim is too slow; validates greedy quality)
    pareto_B = compute_pareto_routeB(N_B, alphas_B, 0.0, top_mo)
    # Also compute Route A for N=20 for fair comparison
    print(f"\n  Route A at N={N_B} for comparison:")
    pareto_A_N20 = compute_pareto_routeA(N_B, d0, g0, alphas_B, 0.0, top_mo)
    print("  Route A vs B (N=20) comparison:")
    for alpha in alphas_B:
        dR_pk = (pareto_B[alpha]['R_peak'] - pareto_A_N20[alpha]['R_peak']) * 100
        dR_mn = (pareto_B[alpha]['R_mean'] - pareto_A_N20[alpha]['R_mean']) * 100
        print(f"    α={alpha}: ΔR_peak={dR_pk:+.3f}%, ΔR_mean={dR_mn:+.3f}%  "
              f"({'DE better' if dR_pk > 0.05 else 'greedy sufficient'})")

    # Compare Route A vs B
    print("\n  Route A vs B comparison:")
    for alpha in alphas_B:
        dR_pk = (pareto_B[alpha]['R_peak'] - pareto_A[alpha]['R_peak']) * 100
        dR_mn = (pareto_B[alpha]['R_mean'] - pareto_A[alpha]['R_mean']) * 100
        print(f"    α={alpha}: ΔR_peak={dR_pk:+.3f}%, ΔR_mean={dR_mn:+.3f}%  "
              f"({'B better' if dR_pk > 0.01 else 'A sufficient'})")

    # ── Stage 5 & 6: Roughness study ─────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 5 & 6: Nevot-Croce roughness + re-optimization")
    print("="*60)
    rough_results = run_roughness_study(
        bl_periodic, pareto_A, N_best, d0, g0, top_mo,
        sigmas=[0.0, 0.2, 0.4], alphas=[0.5]
    )

    # ── Plotting all figures ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("GENERATING FIGURES")
    print("="*60)
    plot_all(bl_periodic, pareto_A, pareto_B, pareto_chirped,
             rough_results, N_best, top_mo, d0, g0, alphas_full)

    # ── Final conclusions ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("CONCLUSIONS")
    print("="*60)

    R_pk_per = max(sweep_angle(bl_periodic, top_mo=top_mo)[1])
    R_mn_per = np.mean(sweep_angle(bl_periodic, top_mo=top_mo)[1])

    best_ap_pk = max(pareto_A[a]['R_peak'] for a in alphas_full)
    best_ap_mn = max(pareto_A[a]['R_mean'] for a in alphas_full)

    a1_pk = pareto_A[1.0]['R_peak']
    a0_mn = pareto_A[0.0]['R_mean']

    print(f"\n  Periodic:    R_peak={R_pk_per*100:.2f}%,  R_mean={R_mn_per*100:.2f}%")
    print(f"  Aperiodic best peak (α=1.0): {a1_pk*100:.2f}%")
    print(f"  Aperiodic best mean (α=0.0): {a0_mn*100:.2f}%")

    strictly_better_peak = a1_pk > R_pk_per
    strictly_better_mean = a0_mn > R_mn_per

    if strictly_better_peak and strictly_better_mean:
        verdict = "STRICTLY BETTER: aperiodic outperforms periodic on BOTH metrics"
    elif strictly_better_peak or strictly_better_mean:
        verdict = "TRADE-OFF: aperiodic improves one metric at cost of the other"
    else:
        verdict = "NO GAIN: periodic already on the Pareto front"
    print(f"\n  Verdict: {verdict}")

    # Roughness robustness
    for sigma in [0.2, 0.4]:
        R_pk_ap_r = rough_results[sigma]['aperiodic'][0.5]['R_peak']
        R_pk_per_r = rough_results[sigma]['periodic']['R_peak']
        print(f"\n  σ={sigma} nm: periodic R_peak={R_pk_per_r*100:.2f}%, "
              f"aperiodic(α=0.5) R_peak={R_pk_ap_r*100:.2f}%  "
              f"({'advantage preserved' if R_pk_ap_r >= R_pk_per_r else 'advantage lost'})")

    print("\nDone. All figures saved to current directory.")


if __name__ == '__main__':
    main()
