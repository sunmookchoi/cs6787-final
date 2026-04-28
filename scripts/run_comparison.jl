#!/usr/bin/env julia

using CS6787Final
using Printf
using Random

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
            elseif item == "no-random"
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

function opt(parsed, key, default)
    return get(parsed, key, default)
end

function opt_int(parsed, key, default)
    return parse(Int, opt(parsed, key, string(default)))
end

function opt_float(parsed, key, default)
    return parse(Float64, opt(parsed, key, string(default)))
end

function main(args)
    parsed = parse_args(args)

    T = opt_int(parsed, "T", 200)
    H = haskey(parsed, "H") ? opt_int(parsed, "H", 0) : default_H(T)
    L = haskey(parsed, "L") ? opt_int(parsed, "L", 0) : default_L(T)
    system_name = lowercase(opt(parsed, "system", "simple"))
    n = opt_int(parsed, "n", 3)
    p = opt_int(parsed, "p", 2)
    rho = opt_float(parsed, "rho", 0.3)
    seed = opt_int(parsed, "seed", 1)
    reps = opt_int(parsed, "reps", 1)

    sys_rng = Random.Xoshiro(seed)
    system = if system_name == "simple"
        make_simple_system(rho=rho)
    elseif system_name == "random"
        make_random_system(n, p; rho=rho, rng=sys_rng)
    else
        error("--system must be simple or random")
    end

    config = ComparisonConfig(
        T = T,
        H = H,
        L = L,
        sigma_w = opt_float(parsed, "sigma-w", 0.01),
        sigma_z = opt_float(parsed, "sigma-z", 0.01),
        sdp_round_restarts = opt_int(parsed, "round-restarts", 128),
        low_rank_rank = opt_int(parsed, "low-rank-rank", 0),
        low_rank_restarts = opt_int(parsed, "low-rank-restarts", 8),
        low_rank_maxiter = opt_int(parsed, "low-rank-maxiter", 1000),
        low_rank_tol = opt_float(parsed, "low-rank-tol", 1e-7),
        low_rank_round_restarts = opt_int(parsed, "low-rank-round-restarts",
            opt_int(parsed, "round-restarts", 128)),
        include_random = !haskey(parsed, "no-random"),
    )

    methods = opt(parsed, "methods", "both")
    output = opt(parsed, "output", joinpath(@__DIR__, "..", "results", "comparison.csv"))

    @printf("system=%s T=%d H=%d L=%d reps=%d methods=%s\n",
        system_name, config.T, config.H, config.L, reps, methods)
    rows = run_comparison(system, config; methods=methods, reps=reps, seed=seed)
    write_results_csv(output, rows)

    @printf("wrote %d rows to %s\n", length(rows), output)
    for row in rows
        @printf("%3s rep=%s status=%s realized=%s rounded_true=%s solve_s=%s\n",
            row["method"], row["rep"], row["status"], row["realized_return"],
            row["rounded_true_half_objective"], row["solve_seconds"])
    end
end

main(ARGS)
