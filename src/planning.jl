function build_M_true(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, Ttail::Integer)
    n, p = size(B)
    @assert size(A) == (n, n)
    @assert size(C) == (p, n)
    @assert Ttail >= 1 "Ttail must be positive"

    block_count = Ttail + 1
    T = promote_type(eltype(A), eltype(B), eltype(C), Float64)
    d = p * block_count
    M = zeros(T, d, d)

    blocks = Vector{Matrix{T}}(undef, Ttail)
    Ak = Matrix{T}(I, n, n)
    for k in 1:Ttail
        blocks[k] = C * Ak * B
        Ak = Ak * A
    end

    @views for i in 1:Ttail
        for j in (i + 1):block_count
            M[((i - 1) * p + 1):(i * p), ((j - 1) * p + 1):(j * p)] .= blocks[j - i]
        end
    end
    return M
end

build_M_true(system::BanditSystem, Ttail::Integer) =
    build_M_true(system.A, system.B, system.C, Ttail)

function build_true_qubo_matrix(system::BanditSystem, Tcommit::Integer)
    M = build_M_true(system, Tcommit)
    W = 0.5 .* Matrix(M .+ M')
    @inbounds for i in axes(W, 1)
        W[i, i] = 0.0
    end
    return W
end
