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

function make_random_system(n::Integer=3, p::Integer=2; rho::Real=0.5,
        rng::AbstractRNG=Random.default_rng())
    @assert n > 0 && p > 0 "n and p must be positive"
    @assert rho > 0 "rho must be positive"

    A = randn(rng, n, n) / sqrt(n)
    radius = maximum(abs.(eigvals(A)))
    @assert radius > 0 "random A has zero spectral radius"
    A .*= Float64(rho) / radius

    B = randn(rng, n, p) / sqrt(n)
    C = randn(rng, p, n) / sqrt(p)
    return BanditSystem(A, B, C)
end
