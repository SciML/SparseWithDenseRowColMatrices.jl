include("shared/testutils.jl")
using Test, LinearAlgebra

# Build a matrix whose sparse part S is singular (a zero row) but whose assembled A is not,
# because the low-rank fill supplies that row. This is the canonical BVP situation (an
# interior stencil that does not by itself pin down a boundary degree of freedom).
function singularS_nonsingularA(n, r; seed = 1)
    rng = MersenneTwister(seed)
    S = sparse(3.0 * I, n, n)
    for i in 1:(n - 1)
        S[i + 1, i] = -1.0
    end
    for j in 1:n            # zero out the first r rows of S → S singular
        for i in 1:r
            S[i, j] = 0.0
        end
    end
    dropzeros!(S)
    F = zeros(r, n)          # dense rows that make A nonsingular
    for i in 1:r
        F[i, i] = 1.0
        F[i, i + r] = 0.3
    end
    A = SparseWithDenseRowColMatrix(S, F; replace = true)
    return A
end

@testset ":auto falls back to augmented when S is singular" begin
    A = singularS_nonsingularA(50, 2)
    @test_throws SingularException factorize(A; strategy = :woodbury, auto_fallback = false)
    F = factorize(A)                       # :auto
    @test F isa SparseWithDenseRowColAugmented
    @test issuccess(F)
    b = randn(50)
    @test relerr(F \ b, Matrix(A) \ b) < 1.0e-9
end

@testset "augmented adjoint & transpose solve" begin
    A = singularS_nonsingularA(50, 2; seed = 3)
    F = factorize(A)
    @test F isa SparseWithDenseRowColAugmented
    M = Matrix(A)
    b = randn(50)
    @test relerr(F' \ b, M' \ b) < 1.0e-9
    @test relerr(transpose(F) \ b, transpose(M) \ b) < 1.0e-9
    x = copy(b)
    ldiv!(F', x)
    @test relerr(x, M' \ b) < 1.0e-9
end

@testset "forced :augmented agrees on a nonsingular S" begin
    A = rand_sparsedense(60, 3; seed = 5)
    M = Matrix(A)
    b = randn(60)
    F = factorize(A; strategy = :augmented)
    @test F isa SparseWithDenseRowColAugmented
    @test relerr(F \ b, M \ b) < 1.0e-9
    B = randn(60, 3)
    @test relerr(F \ B, M \ B) < 1.0e-9
end

@testset "singular A surfaces SingularException" begin
    n, r = 20, 1
    S = spzeros(n, n)                      # A will be rank-deficient
    F = zeros(r, n)
    F[1, 1] = 1.0
    A = SparseWithDenseRowColMatrix(S, F; replace = true)
    @test_throws SingularException factorize(A; strategy = :augmented)
    @test_throws SingularException factorize(A)   # :auto can't rescue a singular A
end

@testset "invalid strategy" begin
    A = rand_sparsedense(20, 1)
    @test_throws ArgumentError factorize(A; strategy = :nonsense)
end
