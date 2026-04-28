using CS6787Final
using LinearAlgebra
using Random
using Test

@testset "core estimation and planning" begin
    rng = Random.Xoshiro(11)
    system = make_simple_system(rho=0.25)
    config = ComparisonConfig(T=40, H=16, L=3, sigma_w=0.0, sigma_z=0.0,
        include_random=false)
    instance = make_planning_instance(system, config; rng=rng)

    @test size(instance.Ghat) == size(instance.Gtrue)
    @test size(instance.S_hat) == size(instance.S_true)
    @test size(instance.S_hat, 1) == 2 * (config.T - config.H + 1)
    @test isfinite(instance.G_rel_error)
end

@testset "low-rank factorization" begin
    rng = Random.Xoshiro(12)
    S = Symmetric(randn(rng, 8, 8))
    S = Matrix(0.5 * (S + S'))
    sol = low_rank_sdp_factorization(S; rank=3, restarts=2, maxiter=80,
        round_restarts=16, rng=rng)

    @test sol.rank == 3
    @test length(sol.x) == 8
    @test all(abs.(sol.x) .== 1)
    @test isfinite(sol.relaxed_value)
    @test isfinite(sol.rounded_value)
    @test all(abs.(sqrt.(sum(abs2, sol.Y; dims=2)) .- 1.0) .< 1e-8)
end

@testset "qubo solvers" begin
    rng = Random.Xoshiro(13)
    system = make_simple_system(rho=0.3)
    W = build_true_qubo_matrix(system, 8)
    M = build_M_true(system, 8)

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

@testset "comparison pipeline" begin
    system = make_simple_system(rho=0.3)
    config = ComparisonConfig(T=36, H=14, L=3, sigma_w=0.01, sigma_z=0.01,
        low_rank_restarts=2, low_rank_maxiter=60, low_rank_round_restarts=8,
        include_random=true)
    rows = run_comparison(system, config; methods=[:low_rank], reps=2, seed=7)

    @test length(rows) == 4
    @test Set(row["method"] for row in rows) == Set(["low_rank", "random_commit"])
    @test all(haskey(row, "realized_return") for row in rows)
end
