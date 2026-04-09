# Scaling Laws for Operator Learning: Resolution as a First-Class Scaling Axis

> Companion code for: *"Scaling Laws Reveal Resolution as a First-Class
> Variable in Operator Learning"* (Perrier, 2026).

## Motivation

Neural operator methods (DeepONet, FNO) promise mesh-independent learning of
PDE solution operators, yet practitioners lack quantitative guidance on how
error scales with dataset size, model capacity, and discretization resolution.

This repository studies **scaling laws** of the form

$$E(N, D, R) \;=\; E_\infty \;+\; a\,N^{-\alpha} \;+\; b\,D^{-\beta} \;+\; c\,R^{-\gamma}$$

where $N$ is the number of training samples, $D$ is the parameter count, and
$R$ is the discretization resolution.

## Central thesis

For operator learning in scientific ML, **discretization resolution is a
first-class scaling axis** — not just an implementation detail. Scaling
exponents, error floors, and cross-resolution transfer behavior jointly
diagnose when resolution-invariant architectures genuinely help, when they
merely shift the error floor, and when discretization-tied baselines suffice.

## Relationship to Paper 1

This work extends *"Scaling-Law Diagnostics for Physics-Informed Machine
Learning"* (Perrier, 2025), which introduced exponent-based diagnostics for
physics priors (PINNs, HNNs) in low-dimensional benchmarks. Paper 2 moves
from finite-dimensional dynamics to **infinite-dimensional operator learning**
and elevates resolution to a central variable.

## Tasks

| Task | PDE family | Input → Output |
|------|-----------|----------------|
| `burgers_operator` | Viscous Burgers | $u_0(x) \to u(x, T)$ |
| `darcy` | Darcy flow | $a(x) \to u(x)$ |
| `diffusion` | Diffusion | $u_0(x) \to u(x, T)$ |

## Models

| Model | Type | Resolution-dependent? |
|-------|------|----------------------|
| `mlp_baseline` | Discretization-tied MLP | Yes — param count scales with $R$ |
| `deeponet` | Branch–trunk DeepONet | No — architecture is mesh-free |
| `fno` | Fourier Neural Operator | No — operates in frequency domain |

## Scaling axes

| Axis | Symbol | Typical grid |
|------|--------|-------------|
| Dataset size | $N$ | 50, 100, 200, 500, 1k, 2k, 5k |
| Model capacity | $D$ | tiny → xlarge (parameter count) |
| Resolution | $R$ | 32, 64, 128, 256, 512 |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                              # smoke tests

# Pilot sweep (single task, small grid)
python scripts/run_sweep.py --task burgers_operator --pilot

# Full pipeline
bash scripts/run_pipeline.sh
```

## Experiment layout

```
runs/
  task=burgers_operator/
    model=mlp_baseline/
      capacity=medium/
        N=500/
          R=64/
            data_seed=11/
              seed=101/
                metrics.json
```

Each run produces a `metrics.json`. Aggregation, scaling-law fitting, and
figure generation follow the same pipeline pattern as Paper 1.

## Repository structure

```
src/scaling_operator_learning/
  tasks/          # PDE data generation (resolution-parameterized)
  models/         # mlp_baseline, deeponet, fno
  training/       # unified train_one_run()
  analysis/       # power-law fitting, bootstrap, cross-resolution
  utils/          # IO helpers
scripts/          # sweep runner, aggregation, analysis, plotting
configs/          # per-task YAML configs
```

## License

MIT
