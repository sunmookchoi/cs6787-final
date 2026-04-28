# CS6787 Final Experiments

Clean Julia repository for the final-project experiments on bilinear observable
bandits. It keeps the minimal pieces from `repos/bilinear-obs-bandits` needed to
compare:

- `sdp_gw`: the current SDP baseline solved with MOSEK, followed by Gaussian
  hyperplane rounding.
- `sign_iteration`: a fixed-point binary baseline using `x <- sign(Wx)`.
- `box_pgd`: projected gradient ascent on `[-1, 1]^d`, reporting both the
  continuous box objective and the rounded binary objective.
- `low_rank`: a low-rank SDP factorization method on the product of spheres,
  followed by the same rounding style.

The old duplicated experiment folders and logs are intentionally not copied.

## Layout

```text
src/
  CS6787Final.jl   Package entry point
  systems.jl       Stable system generators
  estimation.jl    Reward simulation and G estimation
  planning.jl      Toeplitz planning matrix construction
  solvers.jl       SDP, sign-iteration, box PGD, and low-rank solvers
  mosek_sdp.jl     Optional MOSEK SDP baseline
  experiments.jl   Shared comparison pipeline
scripts/
  run_comparison.jl
  run_qubo_benchmark.jl
test/
  runtests.jl
```

## Quick Start

Run the package smoke tests:

```bash
julia --project=. test/runtests.jl
```

Run a small low-rank-only comparison:

```bash
julia --project=. scripts/run_comparison.jl --methods low_rank --T 80 --H 30 --L 4 --reps 2
```

Run the SDP+GW baseline against the low-rank method:

```bash
julia --project=. scripts/run_comparison.jl --methods both --T 200 --reps 5
```

Run the deterministic QUBO benchmark from the report's simple A/B/C system:

```bash
julia --project=. scripts/run_qubo_benchmark.jl \
  --Ts 100:100:2000 \
  --methods sign_iteration,box_pgd,low_rank,sdp_gw \
  --ranks 2,4,6,8 \
  --sdp-max-T 300 \
  --output results/qubo_benchmark.csv
```

This writes the CSV plus SVG plots under `results/plots/`:

- `figure1_quality_vs_T.svg`: final binary objective `x'Wx` vs commit horizon `T`
- `figure2_runtime_vs_T.svg`: wall-clock time vs `T`, log-scale
- `figure3_peak_memory_vs_T.svg`: peak memory in MB vs `T`, log-scale
- `figure4_rank_quality.svg`: LR-SDP objective vs rank for `T=500,1000,1500,2000`
- `figure4_rank_runtime.svg`: LR-SDP runtime vs rank for `T=500,1000,1500,2000`
- `figure5_anytime_T2000.svg`: best binary objective found so far vs wall-clock time

The anytime trace is also written to `results/anytime_trace.csv`. Pass
`--no-anytime` to skip it, or change the fixed horizon/ranks with
`--anytime-T 2000 --anytime-ranks 4,8`.

`sdp_gw` requires `Mosek.jl` somewhere on Julia's load path and a working MOSEK
license. The package itself can load and run low-rank experiments without
MOSEK.

Results are written to `results/comparison.csv` by default. The CSV records the
shared exploration error, each solver's relaxed objective, rounded objective
under the estimated and true commit matrices, realized return, solve time, and
solver status.

## Useful Options

```bash
julia --project=. scripts/run_comparison.jl \
  --system random \
  --n 5 --p 3 --rho 0.5 \
  --T 400 --H 80 --L 6 \
  --sigma-w 0.05 --sigma-z 0.05 \
  --methods low_rank,sdp_gw \
  --low-rank-rank 12 \
  --low-rank-restarts 8 \
  --low-rank-maxiter 1500 \
  --round-restarts 256 \
  --reps 10 \
  --output results/random_system.csv
```

If `--H` or `--L` is omitted, the script uses the schedule from the previous
experiments: `H = round(2000^-0.05 * T^(2/3))` and `L = round(0.75 log(T))`.

## QUBO Benchmark Notes

`run_qubo_benchmark.jl` builds `W = M + M'` directly from the current report's
default system:

```julia
A = diagm([0.3, 0.15, 0.12])
B = [1 0; 0 1; 0.5 0.4]
C = [1 0 0; 0 1 0.3]
```

For the current `B=3x2` and `C=2x3`, each block `CA^kB` is `2x2`. The script
uses the report convention with `T+1` block rows/columns, so the dense QUBO
dimension is `d = 2(T+1)`.

The benchmark records solver wall time, Julia allocated bytes from `@timed`,
and `Sys.maxrss()` as the process peak resident memory. For strict per-case OS
peak RSS, run a single benchmark command through `/usr/bin/time -l` on macOS or
`/usr/bin/time -v` on Linux.

Current QUBO benchmark defaults use `--box-maxiter 1000 --box-tol 1e-5` and
`--low-rank-maxiter 1000 --low-rank-tol 1e-5`.

Unless overridden on the command line, the benchmark uses adaptive restart and
rounding counts:

```text
SDP+GW roundings:       floor(64 * T / 2000)
Sign-Iteration starts:  floor(64 * T / 2000)
Box-PGA starts:         floor(64 * T / 2000)
LR-SDP starts:          floor(64 * r / 2 * T / 2000)
LR-SDP roundings:       same count as SDP+GW roundings
```

The implementation uses `max(1, floor(...))` so very small smoke-test horizons
still run. For the paper sweep `T=100,200,...,2000`, this matches the formulas
above.
