# Aperiodic Mo/Si EUV Multilayers — Computational Study

A Python codebase studying whether **aperiodic** (non-uniform-thickness) Mo/Si multilayer
mirrors can deliver higher reflectivity than conventional periodic Bragg stacks at
**λ = 13.5 nm** (extreme ultraviolet lithography), especially over a wider range of
incident angles around the chief ray angle θ₀ = 6°.

The repository implements:
- a recursive Fresnel / Transfer-Matrix Method (TMM) reflectivity solver,
- two-stage optimization (greedy layer-by-layer growth + coordinate-descent refinement),
- an objective function with a min-penalty term `J = mean(R) + λ · min(R)` that
  prevents dead-angle collapse,
- multistart wrappers to mitigate greedy local minima,
- polarization-aware optimization (TE-only vs averaged),
- and a full window-sweep / parameter-sweep / per-layer plotting suite.

---

## Research Question

> Aperiodic multilayers for EUVL: do they increase reflectivity and will they also be
beneficial for larger swings of the incident angle?

**Short answer (from the experiments here):**

| Window ±Δθ | Aperiodic vs Periodic | Why |
|---|---|---|
| ≤ 6.3° (Bragg plateau) | ≈ tie / slight loss | Periodic is already optimal in plateau |
| 7° – 15° (slope region) | + 4 – 6 pp R_mean, up to + 50 pp R_min | Aperiodic chirps the stack to widen the angular response |

A `min(R)` penalty (`λ ≥ 1`) is essential at large windows to prevent the
optimizer from sacrificing entire angles (otherwise R_min collapses to ~ 2 %).

---

## Repository Layout

```
euvl_multilayer_code/
├── README.md                  # this file
├── code/                      # numerical experiments — each script is self-contained
│   ├── euv_multilayer.py        Initial 6-stage study (±1° window)
│   ├── euv_wideband.py          Multi-window grid (Δθ ∈ {1,3,6,10,15}°, α ∈ {0,0.5,1})
│   ├── euv_sweeps.py            Independent α-sweep / window-sweep
│   ├── euv_mean_only.py         Pure mean-only objective (α = 0)
│   ├── euv_minpenalty.py        New objective: J = mean + λ·min, with multistart
│   ├── euv_best.py              Full grid (window × λ × multistart), best-design selection
│   └── euv_polarization.py      TE-only vs unpolarized objective comparison
├── plotting/                  # post-processing plots from saved pickles
│   ├── plot_layers.py           Per-layer (d_pair, γ, d_Mo, d_Si) profiles
│   ├── plot_best_mean_clean.py  Clean R_mean vs window + R(θ) overlay
│   ├── plot_pol_best_mean.py    Polarized (TE-opt) summary panels
│   ├── plot_avg_best_mean.py    Unpolarized (avg-opt) summary panels
│   └── plot_periodic_only.py    Periodic R_avg(θ) reference curve
└── figures/                   # output PNGs are written here when scripts run
```

---

## Requirements

```
python ≥ 3.9
numpy
scipy            # optimize.minimize (Nelder-Mead) and differential_evolution
matplotlib
```

Install with:

```bash
pip install numpy scipy matplotlib
```

---

## Quick Start

Each `code/*.py` script is **self-contained** (it inlines its own TMM solver
to keep file dependencies minimal). Pick one and run it directly:

```bash
cd code/

# Fastest — sanity-check the TMM solver on a periodic baseline (~30 s)
python3 euv_mean_only.py

# Comprehensive — full window × λ × multistart grid (~25 min)
python3 euv_best.py

# Polarized vs unpolarized comparison (~20 min)
python3 euv_polarization.py
```

The code scripts write a pickled results file (`*_results.pkl`) and several PNG
figures into the working directory. To regenerate plots from saved results, run
the matching script in `plotting/`:

```bash
cd ../plotting/
python3 plot_pol_best_mean.py        # uses ../code/euv_pol_results.pkl
python3 plot_avg_best_mean.py
python3 plot_layers.py
```

> **Note:** the plotting scripts assume the pickle files live in the project
> root (one level up from each subfolder). Either run scripts from the same
> directory as the pickle, or adjust the `ROOT` path inside the script.

---

## Physical Model

| Parameter | Value |
|---|---|
| Wavelength λ | 13.5 nm |
| Chief ray angle θ₀ | 6° (NA = 0.33 EUVL) |
| Number of bilayers N | 40 |
| Mo refractive index | 0.921 + 0.0064 i |
| Si refractive index | 0.999 + 0.0018 i |
| Substrate | semi-infinite Si |
| Per-bilayer DOFs | (d_pair ∈ [4, 14] nm, γ = d_Mo / d_pair ∈ [0.15, 0.70]) |
| Aperiodic total DOFs | 2N = 80 |

Reflectivity is computed by a numerically stable **recursive Fresnel** scheme,
treating s and p polarizations independently:

```
r_i = (r_{i,i+1} + r_{i+1} · e^{2iφ}) / (1 + r_{i,i+1} · r_{i+1} · e^{2iφ})
```

with `φ = k_{z,i+1} · d_{i+1}`. See `code/euv_polarization.py` for the
canonical implementation, including TE / TM / averaged outputs.

---

## Objective Function (evolution)

| Stage | Formula | Notes |
|---|---|---|
| (i) Peak-only | `J = R(θ₀)` | Standard periodic optimization |
| (ii) α-weighted | `J = α R(θ₀) + (1 − α) ⟨R⟩` | Trades peak vs angular average |
| (iii) Min-penalty | `J = ⟨R⟩ + λ · min(R)` | Prevents dead-angle collapse |

The angular **mean** is sampled with 11 points during optimization and 161
points at evaluation time. **R_min** is the worst-case reflectivity over the
window — a critical metric for lithography, since pupil-edge angles cannot be
allowed to collapse.

---

## Optimization Strategy

1. **Periodic baseline**: Nelder–Mead on (d_pair, γ) with 2 free parameters,
   initialized at the Bragg estimate `d ≈ λ / (2 · n_avg · cos θ₀)`. Optimizes
   peak-only. Result: `d ≈ 6.95 nm, γ ≈ 0.40, R_peak ≈ 75 %`.

2. **Aperiodic — Stage 1: greedy layer-by-layer growth** (Yamamoto / Namioka
   '92): grow the stack from top to substrate; for each new layer i ∈ {1, …, N}
   only its (d_i, γ_i) is optimized with all earlier layers frozen.
   Complexity O(N · 2-D opt).

3. **Aperiodic — Stage 2: coordinate-descent refinement**: one pass over all
   layers; each layer is re-optimized with neighbours frozen. Compensates for
   greedy's bias toward the top of the stack.

4. **Multistart (K = 3–4)**: each grid point is optimized from K perturbed
   initial seeds (d_0 scaled by 0.94 – 1.06, γ_0 offset ± 3 %). The best-scoring
   design is retained. Observed J-value spread across the K starts is typically
   10 – 35 percentage points — confirming that single-run greedy is unreliable
   without multistart.

---

## Key Findings

1. **Bragg angular FWHM ≈ 12.5°** → plateau half-width ≈ 6.3°. Inside this
   plateau, periodic mirrors are essentially optimal.

2. **Outside the plateau**, aperiodic stacks can yield up to **+ 6 pp on
   R_mean** and **+ 50 pp on R_min** at ±10° windows.

3. **Pure mean optimization** (α = 0) catastrophically fails at ±15°: R_min
   drops to ~ 2 %. The min-penalty formulation `J = mean + λ · min`
   (typically λ = 1) restores R_min to ~ 50 % with only a small mean cost.

4. **Polarization matters**: optimizing for TE only (matching modern
   polarized-EUV systems) gives ~ + 1 – 2 pp on R_TE_mean over averaged
   optimization. The "central dip" of unpolarized aperiodic curves (visible in
   the literature) is reproduced here as a R_p Brewster-like effect at θ₀.

5. **Greedy local minima are real**: at certain (window, λ) configurations,
   single-start runs land in basins 30 + pp worse than the best multistart.
   Future work should consider chirp parameterization (4 parameters) or global
   optimizers (Differential Evolution / CMA-ES).

---

## Suggested Future Extensions

- **Interface roughness** σ ≈ 0.3 – 0.5 nm via Nevot–Croce factor
  (most realistic next step toward fabrication-aware design).
- **MoSi₂ inter-diffusion layer** (0.5 – 1 nm).
- **Ru capping layer** (industrial standard for oxidation resistance).
- **2 % wavelength bandwidth integration** for true source spectrum.
- **Chirp parameterization** `d_i = d_0 + a · i + b · i²` to reduce 80 DOFs to
  4 and avoid greedy local minima.
- **Differential Evolution / CMA-ES** as a global optimizer baseline.
- **High-NA EUV** at θ₀ = 5.355° with anamorphic optics.

---

## File-by-File Pointer

| File | Purpose |
|---|---|
| `code/euv_multilayer.py` | Initial 6-stage study (TMM validation, Pareto, α-sweep) at ±1° |
| `code/euv_wideband.py` | Demonstrates the plateau vs slope distinction across windows |
| `code/euv_sweeps.py` | Independent α-sweep at fixed window, window-sweep at fixed α |
| `code/euv_mean_only.py` | α = 0 (pure mean) — exposes the R_min collapse problem |
| `code/euv_minpenalty.py` | First introduction of `J = mean + λ·min` + multistart |
| `code/euv_best.py` | Full window × λ × multistart grid, dual selection criteria |
| `code/euv_polarization.py` | TE-opt vs avg-opt — reproduces the polarized aperiodic curves |
| `plotting/plot_layers.py` | Per-layer thickness / γ / d_Mo / d_Si profiles |
| `plotting/plot_best_mean_clean.py` | Clean R_mean and R(θ) panels for paper figures |
| `plotting/plot_pol_best_mean.py` | Polarized aperiodic summary plots |
| `plotting/plot_avg_best_mean.py` | Unpolarized aperiodic summary plots |
| `plotting/plot_periodic_only.py` | Periodic R_avg(θ) reference curve |

---

## License

Code in this repository is provided as research material; use freely. If you
use it in a publication, a citation back to this repository is appreciated.
