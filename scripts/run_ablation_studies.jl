#!/usr/bin/env julia

using CS6787Final
using Dates
using LinearAlgebra
using Printf
using Random

const SOLVER_COLUMNS = [
    "system", "system_seed", "rho_A", "n", "p",
    "T", "d", "method", "method_label", "rank", "seed_index", "seed",
    "status", "binary_objective", "continuous_objective", "relaxed_objective",
    "solve_seconds", "rounding_seconds", "total_seconds", "iterations",
    "best_iteration", "best_restart", "grad_norm", "converged",
    "round_restarts", "maxrss_mb",
]

const ROUNDING_COLUMNS = [
    "system", "system_seed", "rho_A", "n", "p",
    "T", "d", "method", "method_label", "rank", "seed_index", "seed",
    "rounding_index", "sample_objective", "best_objective",
    "rounding_elapsed_seconds",
]

function parse_args(args)
    parsed = Dict{String,String}()
    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--")
            item = arg[3:end]
            if occursin("=", item)
                key, value = split(item, "="; limit=2)
                parsed[key] = value
                i += 1
            elseif item in ("resume", "no-sdp")
                parsed[item] = "true"
                i += 1
            else
                @assert i < length(args) "Missing value for --$(item)"
                parsed[item] = args[i + 1]
                i += 2
            end
        else
            error("Unexpected positional argument: $(arg)")
        end
    end
    return parsed
end

opt(parsed, key, default) = get(parsed, key, default)
opt_int(parsed, key, default) = parse(Int, opt(parsed, key, string(default)))
opt_float(parsed, key, default) = parse(Float64, opt(parsed, key, string(default)))

function timestamp()
    return Dates.format(now(), "HH:MM:SS")
end

function fmt_duration(seconds::Real)
    value = Float64(seconds)
    if !isfinite(value) || value < 0
        return "unknown"
    elseif value < 60
        return @sprintf("%.1fs", value)
    elseif value < 3600
        return @sprintf("%.1fm", value / 60)
    end
    return @sprintf("%.2fh", value / 3600)
end

current_rss_mb() = Sys.maxrss() / 2.0^20

function flush_progress()
    flush(stdout)
    flush(stderr)
    return nothing
end

function method_label(method::Symbol, rank)
    method == :sign_iteration && return "Sign-Iteration"
    method == :box_pgd && return "Box-PGA"
    method == :low_rank && return "LR-SDP r=$(rank)"
    method == :sdp_gw && return "SDP+GW"
    return String(method)
end

function spectral_radius(A::AbstractMatrix)
    return maximum(abs.(eigvals(Matrix(A))))
end

function make_ablation_system(parsed)
    system_name = lowercase(strip(opt(parsed, "system", "random")))
    n = opt_int(parsed, "n", 3)
    p = opt_int(parsed, "p", 2)

    if system_name in ("simple", "current", "deterministic")
        rho = opt_float(parsed, "rho", 0.3)
        system = make_simple_system(rho=rho)
        return system, (
            system = "simple",
            system_seed = missing,
            rho_A = spectral_radius(system.A),
            n = size(system.A, 1),
            p = size(system.B, 2),
        )
    elseif system_name == "random"
        rho = opt_float(parsed, "rho", 0.5)
        system_seed = opt_int(parsed, "system-seed", 1)
        rng = Random.Xoshiro(system_seed)
        system = make_random_system(n, p; rho=rho, rng=rng)
        return system, (
            system = "random",
            system_seed = system_seed,
            rho_A = spectral_radius(system.A),
            n = n,
            p = p,
        )
    end

    error("--system must be simple/current/deterministic or random")
end

function system_row(system_info)
    return Dict{String,Any}(
        "system" => system_info.system,
        "system_seed" => system_info.system_seed,
        "rho_A" => system_info.rho_A,
        "n" => system_info.n,
        "p" => system_info.p,
    )
end

function print_system_matrix(name::AbstractString, M::AbstractMatrix)
    println("[$(timestamp())] $name =")
    show(stdout, "text/plain", Matrix(M))
    println()
    flush_progress()
end

function method_seed(base_seed::Integer, method_idx::Integer, seed_index::Integer;
        rank=missing)
    return base_seed + 1_000_000 * method_idx + 1_000 * seed_index +
        (rank === missing ? 0 : Int(rank))
end

nt_get(nt::NamedTuple, key::Symbol, default) = haskey(nt, key) ? nt[key] : default

function csv_cell(x)
    if x === missing || x === nothing
        return ""
    end
    s = string(x)
    if occursin(",", s) || occursin("\"", s) || occursin("\n", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function init_csv(path::AbstractString, columns; resume::Bool=false)
    mkpath(dirname(path))
    if !(resume && isfile(path))
        open(path, "w") do io
            println(io, join(columns, ","))
        end
    end
end

function append_csv_row(path::AbstractString, columns, row::Dict{String,Any})
    open(path, "a") do io
        println(io, join((csv_cell(get(row, col, missing)) for col in columns), ","))
    end
end

function csv_split(line::AbstractString)
    cells = String[]
    io = IOBuffer()
    inquote = false
    i = firstindex(line)
    while i <= lastindex(line)
        c = line[i]
        if c == '"'
            if inquote && i < lastindex(line) && line[nextind(line, i)] == '"'
                write(io, '"')
                i = nextind(line, i)
            else
                inquote = !inquote
            end
        elseif c == ',' && !inquote
            push!(cells, String(take!(io)))
        else
            write(io, c)
        end
        i = nextind(line, i)
    end
    push!(cells, String(take!(io)))
    return cells
end

function completed_solver_cases(path::AbstractString)
    cases = Set{Tuple{String,String,Int}}()
    isfile(path) || return cases
    lines = readlines(path)
    isempty(lines) && return cases
    cols = csv_split(lines[1])
    idx(name) = findfirst(==(name), cols)
    for line in lines[2:end]
        isempty(strip(line)) && continue
        vals = csv_split(line)
        status = vals[idx("status")]
        startswith(status, "failed:") && continue
        label = vals[idx("method_label")]
        rank = vals[idx("rank")]
        seed_index = parse(Int, vals[idx("seed_index")])
        push!(cases, (label, rank, seed_index))
    end
    return cases
end

function sym_zero_diag(W::AbstractMatrix)
    @assert size(W, 1) == size(W, 2)
    Wsym = 0.5 .* (Float64.(W) .+ Float64.(W)')
    @inbounds for i in axes(Wsym, 1)
        Wsym[i, i] = 0.0
    end
    return Wsym
end

function sign_vector_from_scores(scores::AbstractVector)
    x = Vector{Int8}(undef, length(scores))
    @inbounds for i in eachindex(scores)
        x[i] = scores[i] < 0 ? Int8(-1) : Int8(1)
    end
    return x
end

function qubo_value_float(W::AbstractMatrix, x::AbstractVector)
    xf = Float64.(x)
    return dot(xf, W * xf)
end

function rounding_trace_from_factor(W::AbstractMatrix, Y::AbstractMatrix;
        round_restarts::Integer, rng::AbstractRNG)
    Wsym = sym_zero_diag(W)
    d, r = size(Y)
    scores = Vector{Float64}(undef, d)
    rows = Vector{NamedTuple}(undef, round_restarts)
    best = -Inf
    start_time = time()
    for k in 1:round_restarts
        g = randn(rng, r)
        mul!(scores, Y, g)
        sample = qubo_value_float(Wsym, sign_vector_from_scores(scores))
        best = max(best, sample)
        rows[k] = (
            rounding_index = k,
            sample_objective = sample,
            best_objective = best,
            rounding_elapsed_seconds = time() - start_time,
        )
    end
    return rows
end

function rounding_trace_from_sdp(W::AbstractMatrix, X::AbstractMatrix;
        round_restarts::Integer, rng::AbstractRNG)
    Xsym = 0.5 .* (X .+ X')
    lam, U = eigen(Xsym)
    lam = max.(lam, 0.0)
    keep = lam .> 1e-10
    Y = any(keep) ? U[:, keep] .* sqrt.(lam[keep])' : ones(Float64, size(X, 1), 1)
    return rounding_trace_from_factor(W, Y; round_restarts=round_restarts, rng=rng)
end

function solver_row(T, d, method, rank, seed_index, seed, sol, solve_seconds,
        rounding_seconds, round_restarts)
    return Dict{String,Any}(
        "T" => T,
        "d" => d,
        "method" => String(method),
        "method_label" => method_label(method, rank),
        "rank" => rank,
        "seed_index" => seed_index,
        "seed" => seed,
        "status" => nt_get(sol, :status, "ok"),
        "binary_objective" => nt_get(sol, :binary_value,
            nt_get(sol, :rounded_value, missing)),
        "continuous_objective" => nt_get(sol, :continuous_value, missing),
        "relaxed_objective" => nt_get(sol, :relaxed_value, missing),
        "solve_seconds" => solve_seconds,
        "rounding_seconds" => rounding_seconds,
        "total_seconds" => solve_seconds + rounding_seconds,
        "iterations" => nt_get(sol, :iterations, missing),
        "best_iteration" => nt_get(sol, :best_iteration, missing),
        "best_restart" => nt_get(sol, :best_restart, missing),
        "grad_norm" => nt_get(sol, :grad_norm, missing),
        "converged" => nt_get(sol, :converged, missing),
        "round_restarts" => round_restarts,
        "maxrss_mb" => current_rss_mb(),
    )
end

function failure_row(T, d, method, rank, seed_index, seed, err, elapsed,
        round_restarts)
    return Dict{String,Any}(
        "T" => T,
        "d" => d,
        "method" => String(method),
        "method_label" => method_label(method, rank),
        "rank" => rank,
        "seed_index" => seed_index,
        "seed" => seed,
        "status" => "failed: $(typeof(err)): $(err)",
        "binary_objective" => missing,
        "continuous_objective" => missing,
        "relaxed_objective" => missing,
        "solve_seconds" => elapsed,
        "rounding_seconds" => missing,
        "total_seconds" => elapsed,
        "iterations" => missing,
        "best_iteration" => missing,
        "best_restart" => missing,
        "grad_norm" => missing,
        "converged" => false,
        "round_restarts" => round_restarts,
        "maxrss_mb" => current_rss_mb(),
    )
end

function run_seeded_solver(method, W, rank, rng, parsed)
    if method == :sign_iteration
        return sign_iteration(W; restarts=1,
            maxiter=opt_int(parsed, "sign-maxiter", 200), rng=rng)
    elseif method == :box_pgd
        return box_projected_gradient(W; restarts=1,
            maxiter=opt_int(parsed, "box-maxiter", 1000),
            tol=opt_float(parsed, "box-tol", 1e-5), rng=rng)
    elseif method == :low_rank
        return low_rank_sdp_qubo(W; rank=rank, restarts=1,
            maxiter=opt_int(parsed, "low-rank-maxiter", 1000),
            tol=opt_float(parsed, "low-rank-tol", 1e-5),
            round_restarts=1, rng=rng)
    end
    error("Unknown seeded method: $(method)")
end

function write_rounding_trace!(path, system_info, T, d, method, rank, seed_index,
        seed, trace)
    label = method_label(method, rank)
    for row in trace
        data = system_row(system_info)
        merge!(data, Dict{String,Any}(
            "T" => T,
            "d" => d,
            "method" => String(method),
            "method_label" => label,
            "rank" => rank,
            "seed_index" => seed_index,
            "seed" => seed,
            "rounding_index" => row.rounding_index,
            "sample_objective" => row.sample_objective,
            "best_objective" => row.best_objective,
            "rounding_elapsed_seconds" => row.rounding_elapsed_seconds,
        ))
        append_csv_row(path, ROUNDING_COLUMNS, data)
    end
end

function print_case_progress(label, seed_index, seeds_per_method, row,
        completed, skipped, total_cases, run_start, case_start)
    remaining = max(total_cases - completed - skipped, 0)
    elapsed = time() - run_start
    eta = completed > 0 ? elapsed / completed * remaining : NaN
    @printf("[%s] %-14s seed=%3d/%d obj=% .6e solve=%s round=%s case=%s | done=%d skipped=%d remaining=%d/%d elapsed=%s eta=%s rss=%.1fMB\n",
        timestamp(), label, seed_index, seeds_per_method,
        Float64(row["binary_objective"]),
        fmt_duration(Float64(row["solve_seconds"])),
        fmt_duration(Float64(row["rounding_seconds"])),
        fmt_duration(time() - case_start),
        completed, skipped, remaining, total_cases,
        fmt_duration(elapsed), fmt_duration(eta), current_rss_mb())
    flush_progress()
end

function main(args)
    parsed = parse_args(args)
    T = opt_int(parsed, "T", 2000)
    seeds_per_method = opt_int(parsed, "seeds", 256)
    round_restarts = opt_int(parsed, "round-restarts", 64)
    seed_base = opt_int(parsed, "seed", 1)
    progress_every = opt_int(parsed, "progress-every", 16)
    output_dir = opt(parsed, "output-dir",
        joinpath(@__DIR__, "..", "results", "ablation_studies"))
    include_sdp = !haskey(parsed, "no-sdp")
    resume = haskey(parsed, "resume")
    solver_output = joinpath(output_dir, "solver_results.csv")
    rounding_output = joinpath(output_dir, "rounding_trace.csv")

    init_csv(solver_output, SOLVER_COLUMNS; resume=resume)
    init_csv(rounding_output, ROUNDING_COLUMNS; resume=resume)
    done = resume ? completed_solver_cases(solver_output) :
        Set{Tuple{String,String,Int}}()

    system, system_info = make_ablation_system(parsed)
    matrix_timed = @timed build_true_qubo_matrix(system, T)
    W = matrix_timed.value
    d = size(W, 1)

    seeded_cases = Tuple{Symbol,Any,Int}[
        (:sign_iteration, missing, 1),
        (:box_pgd, missing, 2),
        (:low_rank, 2, 3),
        (:low_rank, 4, 4),
    ]
    total_cases = length(seeded_cases) * seeds_per_method + (include_sdp ? 1 : 0)
    completed = 0
    skipped = 0
    run_start = time()

    @printf("[%s] ablation config: T=%d d=%d seeds=%d round_restarts=%d include_sdp=%s resume=%s\n",
        timestamp(), T, d, seeds_per_method, round_restarts, include_sdp, resume)
    @printf("[%s] system=%s system_seed=%s n=%d p=%d rho_A=%.6f\n",
        timestamp(), system_info.system, string(system_info.system_seed),
        system_info.n, system_info.p, Float64(system_info.rho_A))
    print_system_matrix("A", system.A)
    print_system_matrix("B", system.B)
    print_system_matrix("C", system.C)
    @printf("[%s] built W in %s, output_dir=%s\n",
        timestamp(), fmt_duration(matrix_timed.time), output_dir)
    @printf("[%s] outputs: solver=%s rounding=%s planned_cases=%d existing_completed=%d\n",
        timestamp(), solver_output, rounding_output, total_cases, length(done))
    flush_progress()

    for (method, rank, method_idx) in seeded_cases
        label = method_label(method, rank)
        rank_key = rank === missing ? "" : string(rank)
        method_start = time()
        method_done = 0
        @printf("[%s] start %-14s seeds=%d\n", timestamp(), label, seeds_per_method)
        flush_progress()
        for seed_index in 1:seeds_per_method
            if (label, rank_key, seed_index) in done
                skipped += 1
                if seed_index == 1 || seed_index == seeds_per_method ||
                        seed_index % progress_every == 0
                    @printf("[%s] skip %-14s seed=%3d/%d done=%d skipped=%d/%d\n",
                        timestamp(), label, seed_index, seeds_per_method,
                        completed, skipped, total_cases)
                    flush_progress()
                end
                continue
            end

            seed = method_seed(seed_base, method_idx, seed_index; rank=rank)
            rng = Random.Xoshiro(seed)
            case_start = time()
            try
                timed = @timed run_seeded_solver(method, W, rank, rng, parsed)
                sol = timed.value
                rounding_seconds = 0.0
                row = solver_row(T, d, method, rank, seed_index, seed, sol,
                    timed.time, rounding_seconds,
                    method == :low_rank ? round_restarts : missing)
                merge!(row, system_row(system_info))

                if method == :low_rank
                    round_rng = Random.Xoshiro(seed + 500_000_000)
                    rounding_timed = @timed rounding_trace_from_factor(W, sol.Y;
                        round_restarts=round_restarts, rng=round_rng)
                    rounding_seconds = rounding_timed.time
                    write_rounding_trace!(rounding_output, system_info, T, d,
                        method, rank, seed_index, seed, rounding_timed.value)
                    row["rounding_seconds"] = rounding_seconds
                    row["total_seconds"] = timed.time + rounding_seconds
                    row["binary_objective"] = rounding_timed.value[end].best_objective
                end

                append_csv_row(solver_output, SOLVER_COLUMNS, row)
                completed += 1
                method_done += 1
                if seed_index == 1 || seed_index == seeds_per_method ||
                        seed_index % progress_every == 0
                    print_case_progress(label, seed_index, seeds_per_method, row,
                        completed, skipped, total_cases, run_start, case_start)
                end
            catch err
                row = failure_row(T, d, method, rank, seed_index, seed, err,
                    time() - case_start,
                    method == :low_rank ? round_restarts : missing)
                merge!(row, system_row(system_info))
                append_csv_row(solver_output, SOLVER_COLUMNS, row)
                completed += 1
                method_done += 1
                @warn "seeded case failed" method rank seed_index err
                flush_progress()
            end
            GC.gc()
        end
        @printf("[%s] finished %-14s completed_this_run=%d elapsed=%s rss=%.1fMB\n",
            timestamp(), label, method_done, fmt_duration(time() - method_start),
            current_rss_mb())
        flush_progress()
    end

    if include_sdp
        method = :sdp_gw
        rank = missing
        label = method_label(method, rank)
        seed_index = 1
        if (label, "", seed_index) in done
            skipped += 1
            @printf("[%s] skip %-14s done=%d skipped=%d/%d\n",
                timestamp(), label, completed, skipped, total_cases)
            flush_progress()
        else
            seed = method_seed(seed_base, 5, seed_index; rank=rank)
            rng = Random.Xoshiro(seed)
            case_start = time()
            @printf("[%s] start %-14s MOSEK solve\n", timestamp(), label)
            flush_progress()
            try
                timed = @timed solve_sdp_gw_qubo(W; restarts=1, rng=rng)
                sol = timed.value
                round_rng = Random.Xoshiro(seed + 500_000_000)
                rounding_timed = @timed rounding_trace_from_sdp(W, sol.X;
                    round_restarts=round_restarts, rng=round_rng)
                write_rounding_trace!(rounding_output, system_info, T, d, method,
                    rank, seed_index, seed, rounding_timed.value)
                row = solver_row(T, d, method, rank, seed_index, seed, sol,
                    timed.time, rounding_timed.time, round_restarts)
                merge!(row, system_row(system_info))
                row["binary_objective"] = rounding_timed.value[end].best_objective
                append_csv_row(solver_output, SOLVER_COLUMNS, row)
                completed += 1
                print_case_progress(label, seed_index, 1, row, completed, skipped,
                    total_cases, run_start, case_start)
            catch err
                row = failure_row(T, d, method, rank, seed_index, seed, err,
                    time() - case_start, round_restarts)
                merge!(row, system_row(system_info))
                append_csv_row(solver_output, SOLVER_COLUMNS, row)
                completed += 1
                @warn "sdp case failed" err
                flush_progress()
            end
            GC.gc()
        end
    end

    @printf("[%s] sweep complete: run_done=%d skipped=%d elapsed=%s\n",
        timestamp(), completed, skipped, fmt_duration(time() - run_start))
    @printf("[%s] wrote solver rows to %s\n", timestamp(), solver_output)
    @printf("[%s] wrote rounding rows to %s\n", timestamp(), rounding_output)
    flush_progress()
end

main(ARGS)
