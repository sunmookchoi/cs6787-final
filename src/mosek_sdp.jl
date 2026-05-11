const Mosek = Base.require(Base.PkgId(Base.UUID("6405355b-0ac2-5fba-af84-adbd65488c0e"), "Mosek"))

function lower_tri_triplets(S::AbstractMatrix)
    n = size(S, 1)
    @assert n == size(S, 2)
    ii = Int[]
    jj = Int[]
    vv = Float64[]
    for j in 1:n
        for i in j:n
            push!(ii, i)
            push!(jj, j)
            push!(vv, float(S[i, j]))
        end
    end
    return ii, jj, vv
end

function smat_from_barx(barx::Vector{Float64}, n::Int)
    X = zeros(Float64, n, n)
    idx = 1
    for j in 1:n
        X[j, j] = barx[idx]
        idx += 1
        for i in (j + 1):n
            val = barx[idx]
            X[i, j] = val
            X[j, i] = val
            idx += 1
        end
    end
    return X
end

function max_trSX_diag1_psd_mosek_task(S::AbstractMatrix)
    @assert size(S, 1) == size(S, 2) "S must be square"
    n = size(S, 1)
    Ssym = 0.5 * (S + S')
    barci, barcj, barcval = lower_tri_triplets(Ssym)

    barai = Vector{Vector{Int}}(undef, n)
    baraj = Vector{Vector{Int}}(undef, n)
    baraval = Vector{Vector{Float64}}(undef, n)
    for i in 1:n
        barai[i] = [i]
        baraj[i] = [i]
        baraval[i] = [1.0]
    end

    bkc = fill(Mosek.MSK_BK_FX, n)
    blc = fill(1.0, n)
    buc = fill(1.0, n)
    barvardim = [n]

    objval = NaN
    Xopt = Matrix{Float64}(undef, 0, 0)

    Mosek.maketask() do task
        Mosek.putobjsense(task, Mosek.MSK_OBJECTIVE_SENSE_MAXIMIZE)
        Mosek.appendvars(task, 0)
        Mosek.appendcons(task, n)
        Mosek.putconboundslice(task, 1, n + 1, bkc, blc, buc)
        Mosek.appendbarvars(task, barvardim)

        symS = Mosek.appendsparsesymmat(task, barvardim[1], barci, barcj, barcval)
        Mosek.putbarcj(task, 1, [symS], [1.0])

        for i in 1:n
            symai = Mosek.appendsparsesymmat(task, n, barai[i], baraj[i], baraval[i])
            Mosek.putbaraij(task, i, 1, [symai], [1.0])
        end

        Mosek.optimize(task)
        barx = Mosek.getbarxj(task, Mosek.MSK_SOL_ITR, 1)
        Xopt = smat_from_barx(barx, n)
        objval = dot(Ssym, Xopt)
    end

    return objval, Xopt
end
