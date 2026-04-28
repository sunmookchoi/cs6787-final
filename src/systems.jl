struct BanditSystem{TA<:AbstractMatrix,TB<:AbstractMatrix,TC<:AbstractMatrix}
    A::TA
    B::TB
    C::TC
end

function make_simple_system(; rho::Real=0.3)
    A = Diagonal(Float64[rho, 0.5 * rho, 0.4 * rho])
    B = [1.0 0.0;
         0.0 1.0;
         0.5 0.4]
    C = [1.0 0.0 0.0;
         0.0 1.0 0.3]
    return BanditSystem(A, B, C)
end

function make_random_system(n::Integer, p::Integer; rho::Real=0.3, rng::AbstractRNG=Random.default_rng())
    @assert n > 0 && p > 0 "n and p must be positive"

    A = randn(rng, n, n) / sqrt(n)
    rad = maximum(abs.(eigvals(A))) + eps(Float64)
    A .*= Float64(rho) / rad

    B = randn(rng, n, p) / sqrt(n)
    C = randn(rng, p, n) / sqrt(p)
    return BanditSystem(A, B, C)
end

spectral_radius(A::AbstractMatrix) = maximum(abs.(eigvals(Matrix(A))))

function default_H(T::Integer; eps_exp::Real=-0.05, T_opt::Integer=2000)
    @assert T > 1 "T must be larger than 1"
    return max(2, round(Int, T_opt^eps_exp * T^(2 / 3)))
end

function default_L(T::Integer; scale::Real=0.75)
    @assert T > 1 "T must be larger than 1"
    return max(1, round(Int, scale * log(T)))
end
