include("shared/testutils.jl")
using Test, LinearAlgebra, SparseArrays

@testset "public generic sparse interface" begin
    n, r = 20, 2
    S = sparse(4.0I, n, n)
    for i in 1:(n - 1)
        S[i + 1, i] = -1.0
    end
    G = SparseMatrixCSC{Float64, Int32}(S)
    U, V = randn(n, r), randn(r, n) ./ 10
    A = SparseWithDenseRowColMatrix(G, U, V)
    b = randn(n)

    @test Matrix(A) ≈ Matrix(S) + U * V
    @test factorize(A) \ b ≈ Matrix(A) \ b
    @test recommend_lowrank_peel(G).nnz == nnz(S)

    G2 = SparseMatrixCSC{Float64, Int32}(S)
    G2.nzval .*= 1.2
    F = factorize(A)
    refactor!(F, G2; fill = V)
    @test F \ b ≈ Matrix(SparseWithDenseRowColMatrix(G2, U, V)) \ b
end

@testset "structured least-squares factorization accepts complex RHS" begin
    A = rand_sparsedense(30, 2; seed = 93)
    F = SparseWithDenseRowColLeastSquares(A)
    b = randn(30) .+ im .* randn(30)
    @test F \ b ≈ Matrix(A) \ b
end
