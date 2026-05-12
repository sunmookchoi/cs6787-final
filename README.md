# CS6787 Final QUBO Benchmarks

This repository contains commit-phase QUBO benchmarks used for the CS6787 final
project. It compares scalable alternatives to the SDP+GW baseline on the matrix

```text
maximize    x' W x
subject to  x in {-1, +1}^d
```

where `W` is built from either the report's simple latent-bandit system:

```julia
A = diagm([0.3, 0.15, 0.12])
B = [1 0; 0 1; 0.5 0.4]
C = [1 0 0; 0 1 0.3]
```

or a random system with `n=3`, `p=2`: entries of `A0` are sampled i.i.d. from
`N(0, 1/n)` and then scaled as `A = 0.5 A0 / rho(A0)`, while entries of `B` and
`C` are sampled i.i.d. from `N(0, 1/n)` and `N(0, 1/p)`.

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
                   and reproducible random A/B/C system
  planning.jl      Toeplitz matrix and QUBO construction
  solvers.jl       Sign-Iteration, Box-PGA, LR-SDP, SDP+GW wrappers
  mosek_sdp.jl     Optional MOSEK SDP solve
scripts/
  run_qubo_benchmark.jl
  run_ablation_studies.jl
  plot_ablation_studies.jl
  run_lr_seed_ablation.jl
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

Run the full benchmark on the current deterministic system. By default the
scalable methods run through `T=4000`, while `sdp_gw` is skipped after
`T=2000`.

```bash
julia --project=. scripts/run_qubo_benchmark.jl \
  --system simple \
  --Ts 200:200:4000 \
  --methods sign_iteration,box_pgd,low_rank,sdp_gw \
  --ranks 2,4,6,8 \
  --sdp-max-T 2000 \
  --output results/qubo_simple.csv
```

Run the same benchmark on a reproducible random system:

```bash
julia --project=. scripts/run_qubo_benchmark.jl \
  --system random \
  --Ts 200:200:4000 \
  --methods sign_iteration,box_pgd,low_rank,sdp_gw \
  --ranks 2,4,6,8 \
  --sdp-max-T 2000 \
  --output results/qubo_random.csv
```

When `--system random` is used, `--system-seed` defaults to `1` unless you
override it explicitly.

`sdp_gw` requires `Mosek.jl` on Julia's load path and a working MOSEK license.
The other methods run without MOSEK.

By default, the benchmark writes the requested CSV, SVG plots in a directory
named after the CSV stem, and an anytime trace using the same horizon grid as
`--Ts`. For example, `--output results/qubo_random.csv` writes
`results/qubo_random_anytime_trace.csv` and plots under
`results/qubo_random_plots/`, including one `figure5_anytime_T*.svg` plot for
each anytime horizon. Use `--anytime-T 2000` or `--anytime-T 1000:1000:4000`
to override the anytime horizons, and use `--no-anytime` to skip this extra
benchmark. These files are generated artifacts and are ignored by git.

Run the LR-SDP initialization-seed ablation. This fixes the factorization rank
at `r=4`, runs one fresh initialization at a time, and plots the best binary
objective found after the first `K` initial seeds:

```bash
julia --project=. scripts/run_lr_seed_ablation.jl \
  --system simple \
  --T 2000 \
  --rank 4 \
  --seed-counts 24:24:192 \
  --round-restarts 64 \
  --output results/lr_seed_ablation.csv
```

The script writes `results/lr_seed_ablation.csv` and
`results/lr_seed_ablation.svg`. `--seed-counts 24:24:192` runs 192 one-start
initializations and reports the best value at checkpoints
`K=24,48,...,192`. Use `--seed-counts 64` or `--max-seeds 64` to report every
`K=1,2,...,64`. Use `--Ts 1000,2000,4000` to draw one curve per horizon, or
`--system random` for the default random system with `--system-seed 1`.

Run the T=2000 ablation sweep used for group-size and GW-rounding figures:

```bash
julia --project=. scripts/run_ablation_studies.jl \
  --system random \
  --system-seed 1 \
  --T 2000 \
  --seeds 256 \
  --round-restarts 64 \
  --output-dir results/ablation_studies

julia --project=. scripts/plot_ablation_studies.jl \
  --input-dir results/ablation_studies \
  --output-dir results/ablation_studies
```

The sweep writes `solver_results.csv` and `rounding_trace.csv`. The plotting
script writes summary CSVs plus SVG figures in `results/ablation_studies/`.

For a rounding-restart ablation, pass a range or list to `--round-restarts`:

```bash
julia --project=. scripts/run_lr_seed_ablation.jl \
  --system simple \
  --T 2000 \
  --ablation round \
  --rank 4 \
  --seed-counts 192 \
  --round-restarts 5:5:25 \
  --output results/lr_round_restart_ablation.csv
```

This fixes the seed budget at `K=192`, optimizes LR-SDP once per initial seed,
and rerounds each factor with `5,10,15,20,25` random hyperplanes. The plot
`results/lr_round_restart_ablation.svg` has `round_restarts` on the x-axis.
