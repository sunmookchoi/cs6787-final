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
    W = 0.5 .* Matrix(sym_from_M(build_M_true(system, Tcommit)))
    @inbounds for i in axes(W, 1)
        W[i, i] = 0.0
    end
    return W
end

function build_M_from_Ghat(Ghat::AbstractMatrix, Ttail::Integer)
    p, pL = size(Ghat)
    @assert pL % p == 0 "Ghat must have a multiple of p columns"
    @assert Ttail >= 1 "Ttail must be positive"

    L = pL ÷ p
    block_count = Ttail + 1
    d = p * block_count
    M = zeros(eltype(Ghat), d, d)
    Gblocks = [@view Ghat[:, ((k - 1) * p + 1):(k * p)] for k in 1:L]

    @views for i in 1:Ttail
        for j in (i + 1):block_count
            lag = j - i
            if lag <= L
                M[((i - 1) * p + 1):(i * p), ((j - 1) * p + 1):(j * p)] .= Gblocks[lag]
            end
        end
    end
    return M
end

function sym_from_M(M::AbstractMatrix)
    @assert size(M, 1) == size(M, 2) "M must be square"
    return M + M'
end

function vec_to_actions(x::AbstractVector, p::Integer)
    @assert p > 0
    @assert length(x) % p == 0 "length(x) must be a multiple of p"
    m = length(x) ÷ p
    U = reshape(x, p, m)
    return Matrix(transpose(@view U[:, end:-1:1]))
end
