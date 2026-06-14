include("shared/testutils.jl")
using Test, LinearAlgebra

@testset "factorize selects Woodbury for nonsingular S" begin
    A = rand_sparsedense(60, 4)
    F = factorize(A)
    @test F isa SparseWithDenseRowColWoodbury
    @test issuccess(F)
    @test size(F) == (60, 60)
end

@testset "solve vs dense (dense & selector U, several sizes/ranks)" begin
    for (n, r, sel) in ((50, 1, false), (80, 4, false), (60, 3, true), (120, 6, true))
        A = rand_sparsedense(n, r; selector = sel, seed = n + r)
        M = Matrix(A)
        b = randn(n)
        F = factorize(A)
        @test relerr(F \ b, M \ b) < 1.0e-9
        @test relerr(A \ b, M \ b) < 1.0e-9              # \ on the matrix directly
        # multi-RHS
        B = randn(n, 4)
        @test relerr(F \ B, M \ B) < 1.0e-9
        # ldiv! in place
        x = copy(b)
        ldiv!(F, x)
        @test relerr(x, M \ b) < 1.0e-9
    end
end

@testset "adjoint & transpose solve" begin
    A = rand_sparsedense(70, 3; seed = 99)
    M = Matrix(A)
    b = randn(70)
    F = factorize(A)
    @test relerr(F' \ b, M' \ b) < 1.0e-9
    @test relerr(transpose(F) \ b, transpose(M) \ b) < 1.0e-9
    B = randn(70, 3)
    @test relerr(F' \ B, M' \ B) < 1.0e-9
end

@testset "complex adjoint ≠ transpose" begin
    n, r = 40, 2
    rng = MersenneTwister(3)
    S = sparse((3.0 + 0im) * I, n, n)
    for i in 1:(n - 1)
        S[i + 1, i] = -1.0 + 0.2im
    end
    U = randn(rng, ComplexF64, n, r)
    V = randn(rng, ComplexF64, r, n) .* 0.1
    A = SparseWithDenseRowColMatrix(S, U, V)
    M = Matrix(A)
    b = randn(ComplexF64, n)
    F = factorize(A)
    @test relerr(F \ b, M \ b) < 1.0e-9
    @test relerr(F' \ b, M' \ b) < 1.0e-9
    @test relerr(transpose(F) \ b, transpose(M) \ b) < 1.0e-9
end

@testset "iterative refinement helps when S is far worse conditioned than A" begin
    n, r = 60, 1
    S = sparse(1.0 * I, n, n)
    S[1, 1] = 1.0e-9                      # S nearly singular, κ(S) ≈ 1e9
    for i in 1:(n - 1)
        S[i + 1, i] = -0.1
    end
    U = SelectorMatrix{Float64}(n, r)   # correction lives in row 1
    V = zeros(1, n)
    V[1, 1] = 1.0                       # makes A's (1,1) ≈ 1 → A well conditioned
    A = SparseWithDenseRowColMatrix(S, U, V)
    M = Matrix(A)
    b = randn(n)
    xref = M \ b

    F0 = factorize(A; strategy = :woodbury, refine = 0, auto_fallback = false)
    F2 = factorize(A; strategy = :woodbury, refine = 3, auto_fallback = false)
    e0 = relerr(F0 \ b, xref)
    e2 = relerr(F2 \ b, xref)
    @test e2 ≤ e0                       # refinement does not hurt
    @test e2 < 1.0e-6                     # and recovers accuracy
end
