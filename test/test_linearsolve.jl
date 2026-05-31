include("testutils.jl")
using Test, LinearAlgebra, SparseArrays, LinearSolve

n, r = 60, 3

@testset "default solve picks the caching algorithm" begin
    A = rand_sparsedense(n, r; seed = 21)
    b = randn(n)
    sol = solve(LinearProblem(A, b))
    @test sol.u ≈ Matrix(A) \ b
end

@testset "cache reuse: symbolic cached across value/RHS changes" begin
    A = rand_sparsedense(n, r; seed = 22)
    b = randn(n)
    cache = init(LinearProblem(A, b), SparseWithDenseRowColFactorization())
    s1 = solve!(cache)
    @test s1.u ≈ Matrix(A) \ b

    # new values, SAME sparsity pattern → reuse the cached symbolic analysis (refactor!)
    A2 = SparseWithDenseRowColMatrix(
        SparseMatrixCSC(
            size(A)..., copy(sparsepart(A).colptr), copy(sparsepart(A).rowval),
            sparsepart(A).nzval .* 1.5 .+ 0.01
        ),
        lowrankfactors(A)[1], Matrix(fillpart(A)) .+ 0.2
    )
    cache.A = A2
    s2 = solve!(cache)
    @test s2.u ≈ Matrix(A2) \ b

    # new RHS only → reuse the cached numeric factorization
    b2 = randn(n)
    cache.b = b2
    s3 = solve!(cache)
    @test s3.u ≈ Matrix(A2) \ b2
end

@testset "reuse_symbolic = false rebuilds each solve" begin
    A = rand_sparsedense(n, r; seed = 23)
    b = randn(n)
    cache = init(LinearProblem(A, b), SparseWithDenseRowColFactorization(reuse_symbolic = false))
    @test solve!(cache).u ≈ Matrix(A) \ b
    A2 = SparseWithDenseRowColMatrix(
        SparseMatrixCSC(
            size(A)..., copy(sparsepart(A).colptr), copy(sparsepart(A).rowval),
            sparsepart(A).nzval .* 2
        ),
        lowrankfactors(A)[1], Matrix(fillpart(A))
    )
    cache.A = A2
    @test solve!(cache).u ≈ Matrix(A2) \ b
end

@testset "singular A returns Infeasible, does not throw (KLU parity)" begin
    n = 20
    S = sparse(2.0 * I, n, n)
    S[5, 5] = 0.0                         # singular S, and no fill to fix it → singular A
    A = SparseWithDenseRowColMatrix(S, zeros(n, 1), zeros(1, n))
    b = randn(n)
    sol = solve(LinearProblem(A, b))      # must NOT throw
    @test sol.retcode == ReturnCode.Infeasible
    # also in the cached reuse loop
    cache = init(LinearProblem(A, b), SparseWithDenseRowColFactorization())
    @test solve!(cache).retcode == ReturnCode.Infeasible
end

@testset "QR algorithm: solve, cache reuse, and Infeasible parity" begin
    A = rand_sparsedense(n, r; seed = 41)
    b = randn(n)
    cache = init(LinearProblem(A, b), SparseWithDenseRowColQRFactorization())
    s1 = solve!(cache)
    @test s1.retcode == ReturnCode.Success
    @test s1.u ≈ Matrix(A) \ b

    # new values, same pattern → refactor! re-qrs in the cache
    A2 = SparseWithDenseRowColMatrix(
        SparseMatrixCSC(
            size(A)..., copy(sparsepart(A).colptr), copy(sparsepart(A).rowval),
            sparsepart(A).nzval .* 1.5 .+ 0.01
        ),
        lowrankfactors(A)[1], Matrix(fillpart(A)) .+ 0.2
    )
    cache.A = A2
    @test solve!(cache).u ≈ Matrix(A2) \ b
    # new RHS only
    b2 = randn(n)
    cache.b = b2
    @test solve!(cache).u ≈ Matrix(A2) \ b2

    # singular A → Infeasible, no throw (parity with the LU algorithm)
    nn = 20
    S = sparse(2.0 * I, nn, nn)
    S[5, 5] = 0.0
    Asing = SparseWithDenseRowColMatrix(S, zeros(nn, 1), zeros(1, nn))
    sol = solve(LinearProblem(Asing, randn(nn)), SparseWithDenseRowColQRFactorization())
    @test sol.retcode == ReturnCode.Infeasible
end

@testset "QR algorithm stays accurate on ill-conditioned S" begin
    # Where the default (Woodbury) algorithm would lose accuracy, the opt-in QR algorithm holds.
    nn, rr = 120, 3
    S = spdiagm(0 => fill(4.0, nn), 1 => fill(-1.0, nn - 1), -1 => fill(-1.0, nn - 1))
    for i in 1:rr
        S[i, :] .*= 1.0e-13
    end
    dropzeros!(S)
    Fdense = zeros(rr, nn)
    for i in 1:rr
        Fdense[i, i] = 1.0; Fdense[i, i + rr] = 0.3
    end
    A = SparseWithDenseRowColMatrix(S, Fdense; replace = true)
    b = randn(nn)
    xref = Float64.(big.(Matrix(A)) \ big.(b))
    sol = solve(LinearProblem(A, b), SparseWithDenseRowColQRFactorization())
    @test sol.retcode == ReturnCode.Success
    @test norm(sol.u .- xref) / norm(xref) < 1.0e-12
end

@testset "singular S routes through the augmented fallback under caching" begin
    # S singular (zeroed top rows), A nonsingular via the dense fill rows
    nn, rr = 50, 2
    S = sparse(3.0 * I, nn, nn)
    for i in 1:(nn - 1)
        S[i + 1, i] = -1.0
    end
    for j in 1:nn, i in 1:rr
        S[i, j] = 0.0
    end
    dropzeros!(S)
    F = zeros(rr, nn); for i in 1:rr
        F[i, i] = 1.0; F[i, i + rr] = 0.3
    end
    A = SparseWithDenseRowColMatrix(S, F; replace = true)
    b = randn(nn)
    cache = init(LinearProblem(A, b), SparseWithDenseRowColFactorization())
    @test solve!(cache).u ≈ Matrix(A) \ b
    # refactor with new interior values (same pattern), still singular S → still correct
    A2 = SparseWithDenseRowColMatrix(
        SparseMatrixCSC(size(A.S)..., copy(A.S.colptr), copy(A.S.rowval), A.S.nzval .* 1.3),
        A.U, A.V
    )
    cache.A = A2
    @test solve!(cache).u ≈ Matrix(A2) \ b
end
