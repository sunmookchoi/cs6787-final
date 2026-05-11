# CS6787 Final QUBO Benchmarks

This repository contains the deterministic commit-phase QUBO benchmark used for
the CS6787 final project. It compares scalable alternatives to the SDP+GW
baseline on the matrix

```text
maximize    x' W x
subject to  x in {-1, +1}^d
```

where `W` is built from the report's simple latent-bandit system:

```julia
A = diagm([0.3, 0.15, 0.12])
B = [1 0; 0 1; 0.5 0.4]
C = [1 0 0; 0 1 0.3]
```

The benchmark methods are:

- `sdp_gw`: full SDP relaxation solved with MOSEK, followed by Gaussian
  hyperplane rounding.
- `sign_iteration`: fixed-point binary updates using `x <- sign(Wx)`.
- `box_pgd`: projected gradient ascent on `[-1, 1]^d`, followed by sign
  rounding.
- `low_rank`: low-rank SDP factorization on the product of spheres, followed by
  hyperplane rounding.

The explore-then-commit estimation pipeline is intentionally not included.

## Layout

```text
src/
  CS6787Final.jl   Package entry point
  systems.jl       Deterministic simple A/B/C system
  planning.jl      Toeplitz matrix and QUBO construction
  solvers.jl       Sign-Iteration, Box-PGA, LR-SDP, SDP+GW wrappers
  mosek_sdp.jl     Optional MOSEK SDP solve
scripts/
  run_qubo_benchmark.jl
test/
  runtests.jl
```

## Quick Start

Run the tests:

```bash
julia --project=. test/runtests.jl
```

Run a small smoke benchmark without MOSEK:

```bash
julia --project=. scripts/run_qubo_benchmark.jl \
  --Ts 20,40 \
  --methods sign_iteration,box_pgd,low_rank \
  --ranks 2,4 \
  --no-plots \
  --output results/smoke.csv
```

Run the full benchmark:

```bash
julia --project=. scripts/run_qubo_benchmark.jl \
  --Ts 100:100:2000 \
  --methods sign_iteration,box_pgd,low_rank,sdp_gw \
  --ranks 2,4,6,8 \
  --sdp-max-T 300 \
  --output results/qubo_benchmark.csv
```

`sdp_gw` requires `Mosek.jl` on Julia's load path and a working MOSEK license.
The other methods run without MOSEK.

By default, the benchmark writes `results/qubo_benchmark.csv`,
`results/anytime_trace.csv`, and SVG plots under `results/plots/`. These files
are generated artifacts and are ignored by git.
