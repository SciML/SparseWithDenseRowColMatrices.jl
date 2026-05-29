include("testutils.jl")
using Test, LinearAlgebra, SparseArrays

n, r = 60, 3

@testset "refactor! with raw nzval (values only, same pattern)" begin
    A = rand_sparsedense(n, r; seed = 11)
    F = factorize(A)
    b = randn(n)

    S2 = copy(sparsepart(A))
    S2.nzval .*= 1.6
    S2.nzval .+= 0.01
    A2 = SparseWithDenseRowColMatrix(S2, lowrankfactors(A)...)
    refactor!(F, S2.nzval)
    @test relerr(F \ b, Matrix(A2) \ b) < 1.0e-9
end

@testset "refactor! with SparseMatrixCSC (pattern checked) + new fill" begin
    A = rand_sparsedense(n, r; seed = 12, selector = true)
    F = factorize(A)
    b = randn(n)

    S2 = copy(sparsepart(A))
    S2.nzval .= S2.nzval .* 2 .- 0.05
    newfill = randn(r, n)
    refactor!(F, S2; fill = newfill)
    A2 = SparseWithDenseRowColMatrix(S2, SelectorMatrix{Float64}(n, r), newfill)
    @test relerr(F \ b, Matrix(A2) \ b) < 1.0e-9
end

@testset "refactor! with SparseWithDenseRowColMatrix (dense U updated)" begin
    A = rand_sparsedense(n, r; seed = 13, selector = false)
    F = factorize(A)
    b = randn(n)

    S2 = copy(sparsepart(A)); S2.nzval .*= 1.3
    U2 = Matrix(A.U) .+ 0.2 .* randn(n, r)
    V2 = Matrix(A.V) .+ 0.2 .* randn(r, n)
    A2 = SparseWithDenseRowColMatrix(S2, U2, V2)
    refactor!(F, A2)
    @test relerr(F \ b, Matrix(A2) \ b) < 1.0e-9
end

@testset "Newton-style sequence reusing one factorization" begin
    A = rand_sparsedense(n, r; seed = 14)
    F = factorize(A)
    nzpattern = copy(sparsepart(A).nzval)
    for k in 1:6
        newnz = nzpattern .* (1 + 0.1k) .+ 0.001k
        S2 = SparseMatrixCSC(size(A)..., copy(sparsepart(A).colptr), copy(sparsepart(A).rowval), newnz)
        A2 = SparseWithDenseRowColMatrix(S2, lowrankfactors(A)...)
        refactor!(F, newnz)
        b = randn(n)
        @test relerr(F \ b, Matrix(A2) \ b) < 1.0e-9
    end
end

@testset "update_lowrank!: fixed S, varying low-rank correction" begin
    A = rand_sparsedense(n, r; seed = 41, selector = false)
    F = factorize(A)
    S = sparsepart(A); U0 = lowrankfactors(A)[1]
    # vary only V (S and U fixed) — the common Sherman–Morrison–Woodbury reuse case
    for _ in 1:4
        V2 = randn(r, n)
        update_lowrank!(F; V = V2)
        A2 = SparseWithDenseRowColMatrix(copy(S), U0, V2)
        b = randn(n)
        @test relerr(F \ b, Matrix(A2) \ b) < 1.0e-9
        @test relerr(F \ b, factorize(A2) \ b) < 1.0e-9     # matches a fresh full factorization
    end
    # vary U and V together (still reusing S's factorization)
    U2 = randn(n, r); V3 = randn(r, n)
    update_lowrank!(F; U = U2, V = V3)
    A3 = SparseWithDenseRowColMatrix(copy(S), U2, V3)
    b = randn(n)
    @test relerr(F \ b, Matrix(A3) \ b) < 1.0e-9
end

@testset "update_lowrank! with a SelectorMatrix U" begin
    A = rand_sparsedense(n, r; seed = 42, selector = true)   # selector U (dense top rows)
    F = factorize(A)
    V2 = randn(r, n)
    update_lowrank!(F; V = V2)                                # update the dense rows only
    A2 = SparseWithDenseRowColMatrix(copy(sparsepart(A)), SelectorMatrix{Float64}(n, r), V2)
    b = randn(n)
    @test relerr(F \ b, Matrix(A2) \ b) < 1.0e-9
    @test_throws ArgumentError update_lowrank!(F; U = randn(n, r))   # can't swap selector → dense
end

@testset "refactor! rejects an incompatible U change" begin
    A = rand_sparsedense(40, 2; seed = 9, selector = true)   # selector U
    F = factorize(A)
    Adense = SparseWithDenseRowColMatrix(copy(sparsepart(A)), randn(40, 2), Matrix(fillpart(A)))
    @test_throws ArgumentError refactor!(F, Adense)
end

@testset "factorization is independent of later input mutation" begin
    A = rand_sparsedense(50, 2; seed = 10, selector = false)
    M = Matrix(A)
    b = randn(50)
    F = factorize(A)
    x1 = F \ b
    A.V .+= 10.0                 # mutate the inputs after factorize …
    A.U .+= 5.0
    A.S.nzval .*= 2.0
    x2 = F \ b                   # … F owns copies, so its answer is unchanged
    @test x1 ≈ x2
    @test relerr(x1, M \ b) < 1.0e-9
end

@testset "symbolic cache reuse via lu / lu! / ldiv!" begin
    # Analyze once (cache symbolic), reuse for many solves and for new values (same pattern).
    A = rand_sparsedense(n, r; seed = 31)
    F = lu(A)                                  # analyze (cache symbolic) + numeric factor
    for b in (randn(n), randn(n), randn(n))    # reuse the cached factorization for many RHS
        @test relerr(ldiv!(F, copy(b)), Matrix(A) \ b) < 1.0e-9
    end
    A2 = rand_sparsedense(n, r; seed = 31)     # same pattern, new values
    A2.S.nzval .= A.S.nzval .* 1.5 .+ 0.01
    A2.V .= A.V .+ 0.2
    lu!(F, A2)                                 # == refactor!(F, A2); reuses cached symbolic
    b = randn(n)
    @test relerr(F \ b, Matrix(A2) \ b) < 1.0e-9
    @test relerr(F \ b, lu(A2) \ b) < 1.0e-9     # same answer as a fresh full factorization
end

@testset "mismatched pattern is rejected" begin
    A = rand_sparsedense(n, r; seed = 15)
    F = factorize(A)
    Sbad = copy(sparsepart(A))
    Sbad[n, 1] = 123.0                     # introduces a new structural entry
    @test_throws ArgumentError refactor!(F, Sbad)
    @test_throws DimensionMismatch refactor!(F, randn(length(sparsepart(A).nzval) + 1))
end

@testset "augmented refactor!" begin
    rng = MersenneTwister(20)
    nn, rr = 40, 2
    S = sparse(3.0I, nn, nn)
    for i in 1:(nn - 1)
        S[i + 1, i] = -1.0
    end
    for j in 1:nn, i in 1:rr
        S[i, j] = 0.0
    end
    dropzeros!(S)
    Fl = zeros(rr, nn); for i in 1:rr
        Fl[i, i] = 1.0
    end
    A = SparseWithDenseRowColMatrix(S, Fl; replace = true)
    F = factorize(A)
    @test F isa SparseWithDenseRowColAugmented
    b = randn(nn)

    # change interior values only (pattern of augmented system unchanged)
    S2 = copy(sparsepart(A)); S2.nzval .*= 1.4
    A2 = SparseWithDenseRowColMatrix(S2, A.U, A.V)
    refactor!(F, A2)
    @test relerr(F \ b, Matrix(A2) \ b) < 1.0e-9
end
