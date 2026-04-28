Base.@kwdef struct ComparisonConfig
    T::Int = 200
    H::Int = default_H(T)
    L::Int = default_L(T)
    sigma_w::Float64 = 0.01
    sigma_z::Float64 = 0.01
    sdp_round_restarts::Int = 128
    low_rank_rank::Int = 0
    low_rank_restarts::Int = 8
    low_rank_maxiter::Int = 1000
    low_rank_tol::Float64 = 1e-7
    low_rank_round_restarts::Int = 128
    include_random::Bool = true
end

function _validate_config(config::ComparisonConfig)
    @assert config.L < config.H < config.T "Require L < H < T"
    @assert config.sigma_w >= 0 && config.sigma_z >= 0
    return nothing
end

function make_planning_instance(system::BanditSystem, config::ComparisonConfig;
        rng::AbstractRNG=Random.default_rng())
    _validate_config(config)
    _, p = size(system.B)

    U_exp = rand(rng, [-1.0, 1.0], config.H + 1, p)
    _, R_exp = simulate_rewards(system, U_exp; sigma_w=config.sigma_w,
        sigma_z=config.sigma_z, rng=rng)

    Ghat = estimate_G(U_exp, R_exp, config.L)
    Gtrue = true_G(system, config.L)
    G_error = norm(Ghat - Gtrue)
    G_rel_error = G_error / max(norm(Gtrue), eps(Float64))

    Ttail = config.T - config.H
    S_hat = sym_from_M(build_M_from_Ghat(Ghat, Ttail))
    S_true = sym_from_M(build_M_true(system, Ttail))

    return (
        U_exp = U_exp,
        S_hat = S_hat,
        S_true = S_true,
        Ghat = Ghat,
        Gtrue = Gtrue,
        G_error = G_error,
        G_rel_error = G_rel_error,
        commit_horizon = Ttail,
    )
end

function _solver_result(method::Symbol, S::AbstractMatrix, config::ComparisonConfig,
        rng::AbstractRNG)
    if method == :sdp_gw
        return solve_sdp_gw(S; restarts=config.sdp_round_restarts, rng=rng)
    elseif method == :low_rank
        return low_rank_sdp_factorization(S; rank=config.low_rank_rank,
            restarts=config.low_rank_restarts, maxiter=config.low_rank_maxiter,
            tol=config.low_rank_tol, round_restarts=config.low_rank_round_restarts,
            rng=rng)
    else
        error("Unknown method: $(method)")
    end
end

function _base_row(system::BanditSystem, config::ComparisonConfig, instance, rep::Integer)
    n, p = size(system.B)
    return Dict{String,Any}(
        "rep" => rep,
        "T" => config.T,
        "H" => config.H,
        "L" => config.L,
        "n" => n,
        "p" => p,
        "commit_horizon" => instance.commit_horizon,
        "sigma_w" => config.sigma_w,
        "sigma_z" => config.sigma_z,
        "spectral_radius_A" => spectral_radius(system.A),
        "G_error_fro" => instance.G_error,
        "G_rel_error_fro" => instance.G_rel_error,
    )
end

function _nt_get(nt::NamedTuple, key::Symbol, default)
    return haskey(nt, key) ? nt[key] : default
end

function _evaluate_solution(system::BanditSystem, config::ComparisonConfig, instance,
        method::Symbol, sol::NamedTuple, solve_seconds::Real, rep::Integer,
        rng::AbstractRNG)
    _, p = size(system.B)
    x = sol.x
    U_commit = vec_to_actions(x, p)
    U_full = vcat(instance.U_exp, U_commit)
    _, R_full = simulate_rewards(system, U_full; sigma_w=config.sigma_w,
        sigma_z=config.sigma_z, rng=rng)

    row = _base_row(system, config, instance, rep)
    row["method"] = String(method)
    row["status"] = _nt_get(sol, :status, "ok")
    row["relaxed_half_objective"] = _nt_get(sol, :relaxed_value, missing)
    row["rounded_est_half_objective"] = quadratic_value(instance.S_hat, x)
    row["rounded_true_half_objective"] = quadratic_value(instance.S_true, x)
    row["realized_return"] = sum(R_full)
    row["solve_seconds"] = solve_seconds
    row["rank"] = _nt_get(sol, :rank, missing)
    row["restarts"] = _nt_get(sol, :restarts, missing)
    row["iterations"] = _nt_get(sol, :iterations, missing)
    row["grad_norm"] = _nt_get(sol, :grad_norm, missing)
    row["converged"] = _nt_get(sol, :converged, missing)
    return row
end

function _failure_row(system::BanditSystem, config::ComparisonConfig, instance,
        method::Symbol, err, rep::Integer)
    row = _base_row(system, config, instance, rep)
    row["method"] = String(method)
    row["status"] = "failed: $(typeof(err)): $(err)"
    row["relaxed_half_objective"] = missing
    row["rounded_est_half_objective"] = missing
    row["rounded_true_half_objective"] = missing
    row["realized_return"] = missing
    row["solve_seconds"] = missing
    row["rank"] = missing
    row["restarts"] = missing
    row["iterations"] = missing
    row["grad_norm"] = missing
    row["converged"] = false
    return row
end

function _random_commit_row(system::BanditSystem, config::ComparisonConfig, instance,
        rep::Integer, sample_rng::AbstractRNG, eval_rng::AbstractRNG)
    d = size(instance.S_hat, 1)
    x = random_sign_vector(sample_rng, d)
    sol = (
        x = x,
        status = "sampled",
        relaxed_value = missing,
        rank = missing,
        restarts = missing,
        iterations = missing,
        grad_norm = missing,
        converged = missing,
    )
    return _evaluate_solution(system, config, instance, :random_commit, sol, 0.0, rep, eval_rng)
end

function normalize_methods(methods)
    if methods isa AbstractString
        text = lowercase(strip(methods))
        if text == "both"
            return [:sdp_gw, :low_rank]
        end
        return Symbol.(strip.(split(text, ",")))
    end
    return Symbol.(methods)
end

function run_comparison(system::BanditSystem, config::ComparisonConfig;
        methods=[:sdp_gw, :low_rank], reps::Integer=1, seed::Integer=1)
    _validate_config(config)
    method_syms = normalize_methods(methods)
    rows = Dict{String,Any}[]

    for rep in 1:reps
        instance_rng = Random.Xoshiro(seed + 10_000 * rep)
        instance = make_planning_instance(system, config; rng=instance_rng)

        for (method_idx, method) in enumerate(method_syms)
            solver_rng = Random.Xoshiro(seed + 100_000 * rep + 1_000 * method_idx)
            eval_rng = Random.Xoshiro(seed + 100_000 * rep + 17)
            try
                t0 = time()
                sol = _solver_result(method, instance.S_hat, config, solver_rng)
                solve_seconds = time() - t0
                push!(rows, _evaluate_solution(system, config, instance, method, sol,
                    solve_seconds, rep, eval_rng))
            catch err
                push!(rows, _failure_row(system, config, instance, method, err, rep))
                @warn "Method failed" method err
            end
        end

        if config.include_random
            random_rng = Random.Xoshiro(seed + 100_000 * rep + 99_999)
            eval_rng = Random.Xoshiro(seed + 100_000 * rep + 17)
            push!(rows, _random_commit_row(system, config, instance, rep, random_rng, eval_rng))
        end
    end
    return rows
end

const RESULT_COLUMNS = [
    "rep", "method", "status", "T", "H", "L", "n", "p", "commit_horizon",
    "sigma_w", "sigma_z", "spectral_radius_A", "G_error_fro",
    "G_rel_error_fro", "relaxed_half_objective", "rounded_est_half_objective",
    "rounded_true_half_objective", "realized_return", "solve_seconds", "rank",
    "restarts", "iterations", "grad_norm", "converged",
]

function _csv_cell(x)
    if x === missing || x === nothing
        return ""
    end
    s = string(x)
    if occursin(",", s) || occursin("\"", s) || occursin("\n", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function write_results_csv(path::AbstractString, rows::Vector{Dict{String,Any}})
    dir = dirname(path)
    if !isempty(dir)
        mkpath(dir)
    end
    columns = copy(RESULT_COLUMNS)
    for row in rows
        for key in keys(row)
            if !(key in columns)
                push!(columns, key)
            end
        end
    end

    open(path, "w") do io
        println(io, join(columns, ","))
        for row in rows
            println(io, join((_csv_cell(get(row, col, missing)) for col in columns), ","))
        end
    end
    return path
end
