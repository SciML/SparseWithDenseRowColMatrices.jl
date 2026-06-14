include("shared/testutils.jl")
using Test, LinearAlgebra, SparseArrays

@testset "r = 0 degenerate (pure sparse)" begin
    n = 20
    S = sparse(2.0 * I, n, n)
    for i in 1:(n - 1)
        S[i + 1, i] = -0.5
    end
    A = SparseWithDenseRowColMatrix(S, Matrix{Float64}(undef, n, 0), Matrix{Float64}(undef, 0, n))
    @test denserank(A) == 0
    @test Matrix(A) ≈ Matrix(S)
    b = randn(n)
    F = factorize(A)
    @test relerr(F \ b, Matrix(S) \ b) < 1.0e-9
    @test A * b ≈ S * b
end

@testset "Int32 index type" begin
    n, r = 25, 2
    S = SparseMatrixCSC{Float64, Int32}(sparse(3.0 * I, n, n))
    for i in 1:(n - 1)
        S[i + 1, i] = -1.0
    end
    A = SparseWithDenseRowColMatrix(S, randn(n, r), randn(r, n) .* 0.1)
    b = randn(n)
    @test relerr(factorize(A) \ b, Matrix(A) \ b) < 1.0e-9
end

@testset "ldiv! is allocation-free (refine = 0 and refine = 1)" begin
    for refine in (0, 1), sel in (true, false)
        A = rand_sparsedense(50, 3; selector = sel, seed = 42)
        F = factorize(A; refine = refine)
        b = randn(50)
        ldiv!(F, copy(b))                       # warm up
        @test (@allocated ldiv!(F, b)) == 0
    end
end

@testset "refactor! allocates only the tiny dense pivot" begin
    A = rand_sparsedense(50, 3; seed = 43)
    F = factorize(A)
    nz = copy(sparsepart(A).nzval)
    refactor!(F, nz)                            # warm up
    a = @allocated refactor!(F, nz)
    @test a < 2048                              # O(r) pivot vector + LU wrapper only
end

@testset "complex RHS in-place ldiv! over a real factorization" begin
    A = rand_sparsedense(40, 2; seed = 7)        # real (Float64) factorization
    M = Matrix(A)
    F = factorize(A)
    @test eltype(F) <: Real
    bc = randn(ComplexF64, 40)
    x = copy(bc); ldiv!(F, x)
    @test relerr(x, M \ bc) < 1.0e-9
    xa = copy(bc); ldiv!(F', xa)
    @test relerr(xa, M' \ bc) < 1.0e-9
    Bc = randn(ComplexF64, 40, 3)
    Xc = copy(Bc); ldiv!(F, Xc)
    @test relerr(Xc, M \ Bc) < 1.0e-9

    # augmented (singular S) path with a complex RHS
    S = sparse(3.0 * I, 30, 30)
    for i in 1:29
        S[i + 1, i] = -1.0
    end
    for j in 1:30
        S[1, j] = 0.0
    end
    dropzeros!(S)
    fr = zeros(1, 30); fr[1, 1] = 1.0; fr[1, 2] = 0.3
    Aaug = SparseWithDenseRowColMatrix(S, fr; replace = true)
    Fa = factorize(Aaug)
    @test Fa isa SparseWithDenseRowColAugmented
    bc2 = randn(ComplexF64, 30)
    xa2 = copy(bc2); ldiv!(Fa, xa2)
    @test relerr(xa2, Matrix(Aaug) \ bc2) < 1.0e-9
end

@testset "issuccess / size reflect state" begin
    A = rand_sparsedense(30, 2)
    F = factorize(A)
    @test issuccess(F)
    @test size(F, 1) == 30 == size(F, 2)

    Asing = SparseWithDenseRowColMatrix(
        spzeros(10, 10),
        (z = zeros(1, 10); z[1, 1] = 1.0; z); replace = true
    )
    @test_throws SingularException factorize(Asing)
end

@testset "qr returns the augmented QR factorization" begin
    A = rand_sparsedense(20, 2)
    b = randn(20)
    F = qr(A)
    @test F isa SparseWithDenseRowColQRAugmented
    @test F \ b ≈ Matrix(A) \ b
end
