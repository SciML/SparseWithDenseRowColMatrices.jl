include("testutils.jl")
using Test, SparseArrays, LinearAlgebra, LinearSolve

# A classic arrowhead matrix = diagonal bulk + dense last row + dense last column, which is a
# rank-2 dense-row/col correction: A = Diagonal(d) + U*V with U (n×2), V (2×n).
function arrowhead(n; seed = 1)
    rng = MersenneTwister(seed)
    d = 3 .+ rand(rng, n)
    c = randn(rng, n - 1)               # last column entries (rows 1:n-1)
    row = randn(rng, n - 1)             # last row entries (cols 1:n-1)
    full = Matrix(Diagonal(d))
    full[1:(n - 1), n] .= c
    full[n, 1:(n - 1)] .= row
    S = sparse(Diagonal(d))
    U = zeros(n, 2); U[1:(n - 1), 1] .= c; U[n, 2] = 1.0
    V = zeros(2, n); V[1, n] = 1.0; V[2, 1:(n - 1)] .= row
    return SparseWithDenseRowColMatrix(S, U, V), full
end

# Block/bordered arrow: sparse interior B + dense border (E columns, F rows, C corner) of
# width k, which is a rank-2k correction over the block-diagonal bulk [B 0; 0 C].
function block_arrow(n, k; seed = 1)
    rng = MersenneTwister(seed); m = n - k
    B = sparse(SymTridiagonal(fill(4.0, m), fill(-1.0, m - 1)))
    E = randn(rng, m, k); Fb = randn(rng, k, m); C = randn(rng, k, k) + k * I
    full = [Matrix(B) E; Fb C]
    S = blockdiag(B, sparse(C))
    U = zeros(n, 2k); U[1:m, 1:k] .= E; for j in 1:k
        U[m + j, k + j] = 1.0
    end
    V = zeros(2k, n); for j in 1:k
        V[j, m + j] = 1.0
    end; V[(k + 1):2k, 1:m] .= Fb
    return SparseWithDenseRowColMatrix(S, U, V), full
end

@testset "arrowhead: structure, solve, refactor" begin
    n = 400
    A, full = arrowhead(n)
    b = randn(n)
    @test denserank(A) == 2
    @test Matrix(A) ≈ full
    @test A * b ≈ full * b
    @test relerr(A \ b, full \ b) < 1.0e-9
    F = factorize(A)
    @test F isa SparseWithDenseRowColWoodbury        # diagonal bulk is nonsingular → fast path
    @test relerr(F \ b, full \ b) < 1.0e-9
    # Newton-style refactor: same arrow pattern, new values
    A2, full2 = arrowhead(n; seed = 2)
    refactor!(F, A2)
    @test relerr(F \ b, full2 \ b) < 1.0e-9
end

@testset "detector recommends peeling an arrowhead" begin
    _, full = arrowhead(200)
    rec = recommend_lowrank_peel(sparse(full))
    @test rec.recommended
    @test rec.rank == 2
    @test rec.dense_rows == 1 && rec.dense_cols == 1
end

@testset "block / bordered arrow (border width k → rank 2k)" begin
    for k in (1, 3, 5)
        n = 250
        A, full = block_arrow(n, k)
        b = randn(n)
        @test denserank(A) == 2k
        @test Matrix(A) ≈ full
        @test relerr(A \ b, full \ b) < 1.0e-9
    end
end

@testset "arrowhead through the LinearSolve caching interface" begin
    n = 300
    A, full = arrowhead(n; seed = 7)
    b = randn(n)
    cache = init(LinearProblem(A, b), SparseWithDenseRowColFactorization())
    @test relerr(solve!(cache).u, full \ b) < 1.0e-9
    # new arrow values, same pattern → reuses cached symbolic
    A2, full2 = arrowhead(n; seed = 8)
    cache.A = A2
    @test relerr(solve!(cache).u, full2 \ b) < 1.0e-9
    # default solve also picks it up
    @test relerr(solve(LinearProblem(A, b)).u, full \ b) < 1.0e-9
end
