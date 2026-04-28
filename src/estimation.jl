function simulate_rewards(A, B, C, u_seq; x0=nothing, sigma_w::Real=0.05,
        sigma_z::Real=0.05, rng::AbstractRNG=Random.default_rng())
    n, p = size(B)
    @assert size(C) == (p, n)
    @assert size(u_seq, 2) == p

    total_steps = size(u_seq, 1)
    x = isnothing(x0) ? zeros(Float64, n) : Vector{Float64}(x0)
    X = zeros(Float64, total_steps + 1, n)
    R = zeros(Float64, total_steps)
    X[1, :] = x

    for t in 1:total_steps
        u = @view u_seq[t, :]
        R[t] = dot(u, C * x) + Float64(sigma_z) * randn(rng)
        x = A * x + B * u + Float64(sigma_w) * randn(rng, n)
        X[t + 1, :] = x
    end
    return X, R
end

simulate_rewards(system::BanditSystem, u_seq; kwargs...) =
    simulate_rewards(system.A, system.B, system.C, u_seq; kwargs...)

function stack_last_L(u_seq::AbstractMatrix, t::Integer, L::Integer)
    nrows, p = size(u_seq)
    @assert 0 <= t < nrows "t out of range"
    @assert L >= 1 "L must be positive"
    @assert t >= L - 1 "need t >= L - 1"

    result = Vector{eltype(u_seq)}(undef, p * L)
    @views for k in 0:(L - 1)
        copyto!(result, k * p + 1, u_seq[t + 1 - k, :], 1, p)
    end
    return result
end

function estimate_G(U::AbstractMatrix, R::AbstractVector, L::Integer)
    H = size(U, 1) - 1
    p = size(U, 2)
    @assert L >= 1 && H >= L "Require 1 <= L <= H"
    @assert length(R) >= H + 1 "R must contain rewards r_0 through r_H"

    m = H - L
    d = p^2 * L
    X = Matrix{eltype(U)}(undef, d, m)

    @views for i in 1:m
        u_now = U[L + i + 1, :]
        u_hist = stack_last_L(U, L + i - 1, L)
        X[:, i] = kron(u_hist, u_now)
    end

    R_sub = R[(L + 2):(H + 1)]
    g_vec = X' \ R_sub
    return reshape(g_vec, p, p * L)
end

function true_G(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, L::Integer)
    n, p = size(B)
    @assert size(A) == (n, n)
    @assert size(C) == (p, n)

    T = promote_type(eltype(A), eltype(B), eltype(C), Float64)
    Ak = Matrix{T}(I, n, n)
    blocks = Matrix{T}[]
    for _ in 1:L
        push!(blocks, C * Ak * B)
        Ak = Ak * A
    end
    return hcat(blocks...)
end

true_G(system::BanditSystem, L::Integer) = true_G(system.A, system.B, system.C, L)
