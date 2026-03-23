# Spectral Regularization of Wasserstein Flows

**Paper:** *Spectral Regularization of Wasserstein Flows* — A. Sepúlveda-Jiménez (2026)  
**Preprint DOI:** [10.5281/zenodo.19139483](https://doi.org/10.5281/zenodo.19139483)  
**License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---

## Overview

This repository contains the paper source and reproducible experiments for the SpMMD framework — a unified family of spectrally regularized Wasserstein gradient flows parametrized by a *qualification function* φ, interpolating between the Maximum Mean Discrepancy (MMD) and the χ²-divergence.

The framework extends the DrMMD of Chen et al. (2025) along four axes:

| Axis | What varies | Key result |
|---|---|---|
| **Spectral filter** | Tikhonov → Showalter *m* → spectral cutoff → Landweber | Barrier improves from *O*(λʳ) to *O*(λ^{r∧ν}) |
| **Flow geometry** | Wasserstein, Fisher–Rao, WFR, kinetic, Bregman, Sinkhorn | √*C*_P rate gain for kinetic; 100× speedup for FR when support is covered |
| **Divergence objective** | KL → Hellinger → Rényi → χ² (*f*-divergence family) | Barrier exponent *r*∧ν universal across all *f*; Hellinger valid when KL = +∞ |
| **Categorical structure** | QualFun category → monad → quasi-category → Θ_n-space → ∞-operad | Colimit of Showalter tower = χ²-flow; Postnikov tower connectivity = convergence rate |

---

## Repository structure

```
spectral-drmmd/
├── paper/
│   ├── spectral_drmmd_snpdf       # pdf paper source (Springer Nature sn-jnl)
├── figures/                        # All figures included in the paper
│   ├── filter_shapes.png
│   ├── jax_three_ring_results.png
│   ├── jax_gaussian_barrier.png
│   ├── jax_qualification_comparison.png
│   ├── jax_momentum_comparison.png
│   ├── student_teacher.png
│   ├── jax_adaptive_lam_r0.5.png
│   ├── jax_adaptive_lam_r1.0.png
│   ├── fr_wfr_spmmd_flows.png
│   ├── kinetic_spmmd_flow.png
│   ├── bregman_mirror_flows.png
│   └── sinkhorn_small_n.png
├── spectral_drmmd_experiments.ipynb  # All 9 experiments (JAX/TPU)
└── README.md
```

---

## Requirements

```bash
pip install numpy scipy matplotlib scikit-learn jax[cuda12]
# For CPU-only:
pip install jax
```

Python ≥ 3.9. Experiments were run on a Google TPU v6e; all cells are JIT-compiled
with `jax.jit` and run identically on GPU or CPU (slower on CPU for Experiments 7–9).

---

## Experiments

All nine experiments are self-contained in `spectral_drmmd_experiments.ipynb`.
Run cells sequentially; each experiment prints a completion message and saves its
figure to `figures/`.

| # | Section | Setup | Validates |
|---|---|---|---|
| 1 | §8.1 | Three-ring 2-D target, *N*=300, Gaussian kernel ℓ=0.3 | SpMMD escapes MMD local minima |
| 2 | §8.2 | 1-D Gaussian π=N(0,6), λ swept over [10⁻³, 1] | *O*(λʳ) barrier of Thm 4.1 |
| 3 | §8.3 | Same 1-D Gaussian, four filters, *r*∈{0.5, 1.5} | Qualification-order barrier Thm 3.1 |
| 4 | §8.4 | 2-D Gaussian, Nesterov β=0.9 vs β=0 | *O*(1/*n*²) momentum rate Thm 6.1 |
| 5 | §8.5 | Neural kernel, D=157 dims, student/teacher | Deep-kernel convergence Thm 7.1 |
| 6 | §8.6 | 2-D anisotropic Gaussian, adaptive λ | Schedule tracks optimum (Eq. 4.6) |
| 7 | §9 | 1-D Gaussian, FR / WFR / χ²-proximal flows | Thm 9.3, Thm 9.5, Prop 9.4 |
| 8 | §10 | 2-D ill-conditioned Gaussian π=N(0,diag(10,1)) | √*C*_P kinetic rate Thm 10.3 |
| 9 | §11 | All six geometries + Sinkhorn small-*N*=25 | Shared barrier floor, Prop 9.4 |

---

## Key results at a glance

- **Spectral barrier:** the approximation floor is *O*(λ^{*r*∧ν}) for all flow geometries and all *f*-divergences. Only the exponential rate varies across geometries.
- **Fisher–Rao speedup:** when μ₀ covers π, FR-SpMMD and χ²-proximal converge in ≈25 iterations versus ≈2,000 for the Wasserstein flow — a >100× reduction in iteration count (Experiment 7).
- **Kinetic speedup:** on a target with Poincaré constant *C*_P = 10, the critically-damped kinetic flow achieves the √*C*_P ≈ 3.16× asymptotic rate improvement predicted by Theorem 10.3 (Experiment 8).
- **Sinkhorn stability:** at *N*=25 particles, the Sinkhorn proximal step reduces step-to-step variance while matching the asymptotic rate, at the cost of an *O*(ε) bias that vanishes as ε → 0 (Experiment 9).

---

## Citation

```bibtex
@article{SepulvedaJimenez2026,
  author  = {A. Sep\'{u}lveda-Jim\'{e}nez},
  title   = {Spectral Regularization of {W}asserstein Flows},
  journal = {Preprint},
  year    = {2026},
  doi     = {10.5281/zenodo.19139483}
}
```

The DrMMD paper this work extends:

```bibtex
@article{Chen2025DrMMD,
  author  = {Zonghao Chen and Aratrika Mustafi and Pierre Glaser
             and Anna Korba and Arthur Gretton and Bharath Sriperumbudur},
  title   = {(De)-Regularized {MMD} Gradient Flow},
  journal = {Journal of Machine Learning Research},
  volume  = {26},
  year    = {2025}
}
```

---

## Reproducibility note

All data are synthetically generated; no external datasets are used.
Every random seed is fixed (`numpy` seed 42 / JAX PRNGKey 42 unless otherwise noted
in the notebook cell header). The notebook produces all paper figures deterministically.
See the *Data availability* statement in the paper for per-experiment seed and
hyperparameter specifications.
