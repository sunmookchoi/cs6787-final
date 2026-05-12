using CS6787Final
using LinearAlgebra
using Random
using Test

@testset "qubo solvers" begin
    rng = Random.Xoshiro(13)
    system = make_simple_system(rho=0.3)
    random_system = make_random_system(3, 2; rho=0.5, rng=Random.Xoshiro(14))
    W = build_true_qubo_matrix(system, 8)
    M = build_M_true(system, 8)

    @test size(random_system.A) == (3, 3)
    @test size(random_system.B) == (3, 2)
    @test size(random_system.C) == (2, 3)
    @test maximum(abs.(eigvals(Matrix(random_system.A)))) ≈ 0.5
    @test size(W) == (18, 18)
    @test all(diag(W) .== 0.0)
    @test W ≈ 0.5 .* (M .+ M')

    x = random_sign_vector(rng, size(W, 1))
    @test qubo_value(W, x) ≈ dot(Float64.(x), W * Float64.(x))

    sign_sol = sign_iteration(W; restarts=2, maxiter=10, rng=rng)
    @test length(sign_sol.x) == size(W, 1)
    @test all(abs.(sign_sol.x) .== 1)
    @test isfinite(sign_sol.binary_value)

    sign_trace = sign_iteration(W; restarts=1, maxiter=5, rng=rng,
        trace=true)
    @test length(sign_trace.trace_time) == length(sign_trace.trace_value)
    @test !isempty(sign_trace.trace_value)

    box_sol = box_projected_gradient(W; restarts=2, maxiter=20, rng=rng)
    @test length(box_sol.x) == size(W, 1)
    @test all(abs.(box_sol.x) .== 1)
    @test all(abs.(box_sol.z) .<= 1.0 + 1e-10)
    @test isfinite(box_sol.continuous_value)
    @test isfinite(box_sol.binary_value)

    box_trace = box_projected_gradient(W; restarts=1, maxiter=5, rng=rng,
        trace=true)
    @test length(box_trace.trace_time) == length(box_trace.trace_value)
    @test !isempty(box_trace.trace_value)

    lr_sol = low_rank_sdp_qubo(W; rank=3, restarts=2, maxiter=20,
        round_restarts=8, rng=rng)
    @test lr_sol.rank == 3
    @test length(lr_sol.x) == size(W, 1)
    @test all(abs.(lr_sol.x) .== 1)
    @test isfinite(lr_sol.relaxed_value)
    @test isfinite(lr_sol.binary_value)
    @test all(abs.(sqrt.(sum(abs2, lr_sol.Y; dims=2)) .- 1.0) .< 1e-8)

    lr_trace = low_rank_sdp_qubo(W; rank=3, restarts=1, maxiter=5,
        round_restarts=4, trace=true, trace_round_restarts=1, rng=rng)
    @test length(lr_trace.trace_time) == length(lr_trace.trace_value)
    @test !isempty(lr_trace.trace_value)
end
