const _MOSEK_SDP_INCLUDED = Ref(false)

function _ensure_mosek_sdp_loaded!()
    if !_MOSEK_SDP_INCLUDED[]
        include(joinpath(@__DIR__, "mosek_sdp.jl"))
        _MOSEK_SDP_INCLUDED[] = true
    end
    return nothing
end

function qubo_value(W::AbstractMatrix, x::AbstractVector)
    xf = Float64.(x)
    return dot(xf, W * xf)
end

function _sym_zero_diag(W::AbstractMatrix)
    @assert size(W, 1) == size(W, 2) "W must be square"
    d = size(W, 1)
    Wsym = Matrix{Float64}(undef, d, d)
    @inbounds for j in 1:d
        for i in 1:d
            Wsym[i, j] = 0.5 * (Float64(W[i, j]) + Float64(W[j, i]))
        end
    end
    @inbounds for i in 1:d
        Wsym[i, i] = 0.0
    end
    return Wsym
end

function _operator_norm_bound(W::AbstractMatrix)
    return max(opnorm(W, Inf), eps(Float64))
end

function _quadratic_with_work!(work::AbstractVector, W::AbstractMatrix,
        x::AbstractVector)
    mul!(work, W, x)
    return dot(x, work)
end

function _factor_objective_with_work!(WY::AbstractMatrix, W::AbstractMatrix,
        Y::AbstractMatrix)
    mul!(WY, W, Y)
    return dot(Y, WY)
end

function random_sign_vector(rng::AbstractRNG, d::Integer)
    x = Vector{Int8}(undef, d)
    @inbounds for i in 1:d
        x[i] = rand(rng, Bool) ? Int8(1) : Int8(-1)
    end
    return x
end

function sign_vector(v::AbstractVector)
    x = Vector{Int8}(undef, length(v))
    @inbounds for i in eachindex(v)
        x[i] = v[i] < 0 ? Int8(-1) : Int8(1)
    end
    return x
end

function _copy_signs_to_float!(dest::AbstractVector{Float64}, src::AbstractVector)
    @assert length(dest) == length(src)
    @inbounds for i in eachindex(src)
        dest[i] = src[i] < 0 ? -1.0 : 1.0
    end
    return dest
end

function _float_to_sign_vector(v::AbstractVector)
    x = Vector{Int8}(undef, length(v))
    @inbounds for i in eachindex(v)
        x[i] = v[i] < 0 ? Int8(-1) : Int8(1)
    end
    return x
end

function _trace_point!(times::Vector{Float64}, values::Vector{Float64},
        start_time::Float64, value::Real)
    val = Float64(value)
    if isfinite(val)
        push!(times, time() - start_time)
        push!(values, val)
    end
    return nothing
end

function _hyperplane_round_qubo_prepared(W::AbstractMatrix, Y::AbstractMatrix;
        restarts::Integer=128, rng::AbstractRNG=Random.default_rng())
    d, r = size(Y)
    @assert size(W) == (d, d)
    @assert restarts >= 1

    scores = Vector{Float64}(undef, d)
    x_float = Vector{Float64}(undef, d)
    work = Vector{Float64}(undef, d)
    best_x = sign_vector(@view Y[:, 1])
    _copy_signs_to_float!(x_float, best_x)
    best_val = _quadratic_with_work!(work, W, x_float)

    for _ in 1:restarts
        g = randn(rng, r)
        mul!(scores, Y, g)
        _copy_signs_to_float!(x_float, scores)
        val = _quadratic_with_work!(work, W, x_float)
        if val > best_val
            best_x = _float_to_sign_vector(x_float)
            best_val = val
        end
    end
    return best_x, best_val
end

function hyperplane_round_qubo(W::AbstractMatrix, Y::AbstractMatrix;
        restarts::Integer=128, rng::AbstractRNG=Random.default_rng())
    Wsym = _sym_zero_diag(W)
    return _hyperplane_round_qubo_prepared(Wsym, Y; restarts=restarts, rng=rng)
end

function _random_factor(d::Integer, rank::Integer, rng::AbstractRNG)
    Y = randn(rng, d, rank)
    _normalize_rows!(Y, rng)
    return Y
end

function _normalize_rows!(Y::AbstractMatrix, rng::AbstractRNG)
    for i in axes(Y, 1)
        nrm = norm(@view Y[i, :])
        if nrm <= eps(Float64)
            @views Y[i, :] .= randn(rng, size(Y, 2))
            nrm = norm(@view Y[i, :])
        end
        @views Y[i, :] ./= nrm
    end
    return Y
end

function sign_iteration(W::AbstractMatrix; restarts::Integer=32,
        maxiter::Integer=200, rng::AbstractRNG=Random.default_rng(),
        trace::Bool=false, trace_interval::Integer=1)
    Wsym = _sym_zero_diag(W)
    d = size(Wsym, 1)
    @assert restarts >= 1
    @assert maxiter >= 1
    @assert trace_interval >= 1

    work = Vector{Float64}(undef, d)
    x = Vector{Float64}(undef, d)
    x_trial = Vector{Float64}(undef, d)
    best_x = Vector{Int8}(undef, d)
    best_value = -Inf
    best_iter = 0
    best_restart = 0
    total_iterations = 0
    trace_times = Float64[]
    trace_values = Float64[]
    start_time = time()

    for restart in 1:restarts
        _copy_signs_to_float!(x, random_sign_vector(rng, d))
        val = _quadratic_with_work!(work, Wsym, x)
        if val > best_value
            best_x = _float_to_sign_vector(x)
            best_value = val
            best_iter = 0
            best_restart = restart
        end
        trace && _trace_point!(trace_times, trace_values, start_time, best_value)

        iter_done = 0
        for iter in 1:maxiter
            mul!(work, Wsym, x)
            same = true
            @inbounds for i in 1:d
                x_trial[i] = work[i] < 0 ? -1.0 : 1.0
                same &= x_trial[i] == x[i]
            end

            val = _quadratic_with_work!(work, Wsym, x_trial)
            iter_done = iter
            if val > best_value
                best_x = _float_to_sign_vector(x_trial)
                best_value = val
                best_iter = iter
                best_restart = restart
            end
            if trace && (iter == 1 || iter % trace_interval == 0)
                _trace_point!(trace_times, trace_values, start_time, best_value)
            end
            same && break
            x, x_trial = x_trial, x
        end
        total_iterations += iter_done
    end

    return (
        method = :sign_iteration,
        x = best_x,
        rounded_value = best_value,
        binary_value = best_value,
        relaxed_value = missing,
        continuous_value = missing,
        best_continuous_value = missing,
        rank = missing,
        restarts = restarts,
        iterations = total_iterations,
        best_iteration = best_iter,
        best_restart = best_restart,
        grad_norm = missing,
        converged = missing,
        trace_time = trace_times,
        trace_value = trace_values,
        status = "ok",
    )
end

function _round_box_candidate(W::AbstractMatrix, z::AbstractVector)
    x = sign_vector(z)
    xf = Vector{Float64}(undef, length(z))
    _copy_signs_to_float!(xf, x)
    work = similar(z, Float64)
    return x, _quadratic_with_work!(work, W, xf)
end

function box_projected_gradient(W::AbstractMatrix; restarts::Integer=16,
        maxiter::Integer=1000, step_size=nothing, tol::Real=1e-7,
        rng::AbstractRNG=Random.default_rng(), trace::Bool=false,
        trace_interval::Integer=1)
    Wsym = _sym_zero_diag(W)
    d = size(Wsym, 1)
    @assert restarts >= 1
    @assert maxiter >= 1
    @assert trace_interval >= 1

    base_step = isnothing(step_size) ? 0.25 / _operator_norm_bound(Wsym) :
        Float64(step_size)
    work = Vector{Float64}(undef, d)
    grad = Vector{Float64}(undef, d)
    z = Vector{Float64}(undef, d)
    z_trial = Vector{Float64}(undef, d)
    best_z = Vector{Float64}(undef, d)
    best_x = Vector{Int8}(undef, d)
    best_rounded = -Inf
    best_cont_for_round = -Inf
    best_continuous = -Inf
    best_iter = 0
    best_restart = 0
    best_grad_norm = Inf
    total_iterations = 0
    converged_count = 0
    trace_times = Float64[]
    trace_values = Float64[]
    start_time = time()

    for restart in 1:restarts
        rand!(rng, z)
        @. z = 2.0 * z - 1.0
        val = _quadratic_with_work!(work, Wsym, z)
        step = base_step
        iter_done = 0
        grad_norm = Inf
        converged = false

        if trace
            x_round, rounded = _round_box_candidate(Wsym, z)
            if rounded > best_rounded
                copyto!(best_z, z)
                best_x = x_round
                best_rounded = rounded
                best_cont_for_round = val
                best_iter = 0
                best_restart = restart
                best_grad_norm = grad_norm
            end
            _trace_point!(trace_times, trace_values, start_time, best_rounded)
        end

        for iter in 1:maxiter
            mul!(grad, Wsym, z)
            @. grad = 2.0 * grad
            grad_norm = norm(grad) / sqrt(d)

            accepted = false
            trial_step = step
            trial_val = val
            for _ in 1:25
                @inbounds for i in 1:d
                    z_trial[i] = clamp(z[i] + trial_step * grad[i], -1.0, 1.0)
                end
                trial_val = _quadratic_with_work!(work, Wsym, z_trial)
                if trial_val >= val - 1e-12
                    accepted = true
                    break
                end
                trial_step *= 0.5
            end

            if !accepted
                iter_done = iter
                break
            end

            improvement = trial_val - val
            z, z_trial = z_trial, z
            val = trial_val
            step = min(1.05 * trial_step, 10 * base_step)
            iter_done = iter

            if abs(improvement) <= Float64(tol) * max(1.0, abs(val))
                converged = true
                break
            end

            if trace && (iter == 1 || iter % trace_interval == 0)
                x_round, rounded = _round_box_candidate(Wsym, z)
                if rounded > best_rounded
                    copyto!(best_z, z)
                    best_x = x_round
                    best_rounded = rounded
                    best_cont_for_round = val
                    best_iter = iter_done
                    best_restart = restart
                    best_grad_norm = grad_norm
                end
                _trace_point!(trace_times, trace_values, start_time, best_rounded)
            end
        end

        total_iterations += iter_done
        converged_count += converged ? 1 : 0
        if val > best_continuous
            best_continuous = val
        end

        x_round, rounded = _round_box_candidate(Wsym, z)
        if rounded > best_rounded
            copyto!(best_z, z)
            best_x = x_round
            best_rounded = rounded
            best_cont_for_round = val
            best_iter = iter_done
            best_restart = restart
            best_grad_norm = grad_norm
        end
        trace && _trace_point!(trace_times, trace_values, start_time, best_rounded)
    end

    return (
        method = :box_pgd,
        x = best_x,
        z = best_z,
        relaxed_value = best_cont_for_round,
        continuous_value = best_cont_for_round,
        best_continuous_value = best_continuous,
        rounded_value = best_rounded,
        binary_value = best_rounded,
        rank = missing,
        restarts = restarts,
        iterations = total_iterations,
        best_iteration = best_iter,
        best_restart = best_restart,
        grad_norm = best_grad_norm,
        converged = converged_count > 0,
        trace_time = trace_times,
        trace_value = trace_values,
        status = "ok",
    )
end

function low_rank_sdp_qubo(W::AbstractMatrix; rank::Integer=10,
        restarts::Integer=8, maxiter::Integer=1000, step_size=nothing,
        tol::Real=1e-7, round_restarts::Integer=128,
        rng::AbstractRNG=Random.default_rng(), trace::Bool=false,
        trace_interval::Integer=1, trace_round_restarts::Integer=1)
    Wsym = _sym_zero_diag(W)
    d = size(Wsym, 1)
    @assert rank >= 1
    @assert restarts >= 1
    @assert maxiter >= 1
    @assert trace_interval >= 1
    @assert trace_round_restarts >= 1

    r = min(rank, d)
    base_step = isnothing(step_size) ? 0.25 / _operator_norm_bound(Wsym) :
        Float64(step_size)
    WY = Matrix{Float64}(undef, d, r)
    WY_trial = Matrix{Float64}(undef, d, r)
    Y_trial = Matrix{Float64}(undef, d, r)
    best_Y = Matrix{Float64}(undef, 0, 0)
    best_relaxed = -Inf
    best_iter = 0
    best_restart = 0
    best_grad_norm = Inf
    best_converged = false
    total_iterations = 0
    trace_times = Float64[]
    trace_values = Float64[]
    trace_best = -Inf
    trace_rng = Random.Xoshiro(987654321)
    start_time = time()

    for restart in 1:restarts
        Y = _random_factor(d, r, rng)
        val = _factor_objective_with_work!(WY, Wsym, Y)
        step = base_step
        iter_done = 0
        grad_norm = Inf
        converged = false

        if trace
            _, rounded = _hyperplane_round_qubo_prepared(Wsym, Y;
                restarts=trace_round_restarts, rng=trace_rng)
            trace_best = max(trace_best, rounded)
            _trace_point!(trace_times, trace_values, start_time, trace_best)
        end

        for iter in 1:maxiter
            mul!(WY, Wsym, Y)
            grad_norm = 2.0 * norm(WY) / sqrt(length(WY))

            accepted = false
            trial_step = step
            trial_val = val

            for _ in 1:25
                @. Y_trial = Y + 2.0 * trial_step * WY
                _normalize_rows!(Y_trial, rng)
                trial_val = _factor_objective_with_work!(WY_trial, Wsym, Y_trial)
                if trial_val >= val - 1e-12
                    accepted = true
                    break
                end
                trial_step *= 0.5
            end

            if !accepted
                iter_done = iter
                break
            end

            improvement = trial_val - val
            Y, Y_trial = Y_trial, Y
            val = trial_val
            step = min(1.05 * trial_step, 10 * base_step)
            iter_done = iter

            if abs(improvement) <= Float64(tol) * max(1.0, abs(val))
                converged = true
                break
            end

            if trace && (iter == 1 || iter % trace_interval == 0)
                _, rounded = _hyperplane_round_qubo_prepared(Wsym, Y;
                    restarts=trace_round_restarts, rng=trace_rng)
                trace_best = max(trace_best, rounded)
                _trace_point!(trace_times, trace_values, start_time, trace_best)
            end
        end

        total_iterations += iter_done
        if val > best_relaxed
            best_Y = copy(Y)
            best_relaxed = val
            best_iter = iter_done
            best_restart = restart
            best_grad_norm = grad_norm
            best_converged = converged
        end
    end

    x, rounded = _hyperplane_round_qubo_prepared(Wsym, best_Y;
        restarts=round_restarts, rng=rng)
    if trace
        trace_best = max(trace_best, rounded)
        _trace_point!(trace_times, trace_values, start_time, trace_best)
    end
    return (
        method = :low_rank,
        x = x,
        Y = best_Y,
        relaxed_value = best_relaxed,
        continuous_value = best_relaxed,
        best_continuous_value = best_relaxed,
        rounded_value = rounded,
        binary_value = rounded,
        rank = r,
        restarts = restarts,
        round_restarts = round_restarts,
        iterations = total_iterations,
        best_iteration = best_iter,
        best_restart = best_restart,
        grad_norm = best_grad_norm,
        converged = best_converged,
        trace_time = trace_times,
        trace_value = trace_values,
        status = best_converged ? "converged" : "maxiter_or_stalled",
    )
end

function _sdp_hyperplane_round_qubo(W::AbstractMatrix, X::AbstractMatrix;
        restarts::Integer=128, rng::AbstractRNG=Random.default_rng())
    d = size(X, 1)
    Xsym = 0.5 .* (X .+ X')
    lam, U = eigen(Xsym)
    lam = max.(lam, 0.0)
    keep = lam .> 1e-10

    if !any(keep)
        x = random_sign_vector(rng, d)
        return x, qubo_value(W, x)
    end

    V = U[:, keep] .* sqrt.(lam[keep])'
    return _hyperplane_round_qubo_prepared(W, V; restarts=restarts, rng=rng)
end

function solve_sdp_gw_qubo(W::AbstractMatrix; restarts::Integer=128,
        rng::AbstractRNG=Random.default_rng())
    _ensure_mosek_sdp_loaded!()
    Wsym = _sym_zero_diag(W)
    sdp_obj, X = Base.invokelatest(max_trSX_diag1_psd_mosek_task, Wsym)
    x, rounded = _sdp_hyperplane_round_qubo(Wsym, X; restarts=restarts, rng=rng)
    return (
        method = :sdp_gw,
        x = x,
        X = X,
        relaxed_value = sdp_obj,
        continuous_value = sdp_obj,
        best_continuous_value = sdp_obj,
        rounded_value = rounded,
        binary_value = rounded,
        rank = missing,
        restarts = restarts,
        round_restarts = restarts,
        iterations = missing,
        grad_norm = missing,
        converged = true,
        status = "solved",
    )
end
