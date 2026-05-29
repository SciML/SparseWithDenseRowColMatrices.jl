include("testutils.jl")
using Test

n, r = 30, 3

@testset "general A = S + U*V construction" begin
    A = rand_sparsedense(n, r)
    @test A isa SparseWithDenseRowColMatrix
    @test size(A) == (n, n)
    @test eltype(A) == Float64
    @test denserank(A) == r
    @test sparsepart(A) === A.S
    @test fillpart(A) === A.V
    @test lowrankfactors(A) == (A.U, A.V)

    # getindex equals the dense reference everywhere
    M = Matrix(A.S) + Matrix(A.U) * Matrix(A.V)
    @test Matrix(A) ≈ M
    @test all(A[i, j] ≈ M[i, j] for i in 1:n, j in 1:n)
end

@testset "eltype promotion" begin
    S = sparse(2I, 5, 5)              # SparseMatrixCSC{Int}
    U = ones(Float64, 5, 1)
    V = ones(Float64, 1, 5)
    A = SparseWithDenseRowColMatrix(S, U, V)
    @test eltype(A) == Float64
    @test sparsepart(A) isa SparseMatrixCSC{Float64}
    @test eltype(A * ones(5)) == Float64
end

@testset "size / shape validation" begin
    S = sparse(1.0I, 5, 5)
    @test_throws DimensionMismatch SparseWithDenseRowColMatrix(S, ones(4, 1), ones(1, 5))
    @test_throws DimensionMismatch SparseWithDenseRowColMatrix(S, ones(5, 1), ones(1, 4))
    @test_throws DimensionMismatch SparseWithDenseRowColMatrix(S, ones(5, 2), ones(1, 5))
    @test_throws ArgumentError SparseWithDenseRowColMatrix(sparse(1.0I, 5, 6), ones(5, 1), ones(1, 6))
end

@testset "SelectorMatrix convenience (fill)" begin
    S = sparse(2.0I, n, n)
    @inbounds for i in 1:(n - 1)
        S[i + 1, i] = -0.5
    end
    F = randn(r, n)

    Aadded = SparseWithDenseRowColMatrix(S, F; replace = false)
    @test Aadded.U isa SelectorMatrix
    @test fillpart(Aadded) == F
    Madd = Matrix(S)
    Madd[1:r, :] .+= F
    @test Matrix(Aadded) ≈ Madd

    Arep = SparseWithDenseRowColMatrix(S, F; replace = true)
    Mrep = Matrix(S)
    Mrep[1:r, :] = F
    @test Matrix(Arep) ≈ Mrep
    @test Matrix(Arep)[1:r, :] ≈ F
    @test Matrix(Arep)[(r + 1):n, :] ≈ Matrix(S)[(r + 1):n, :]
end

@testset "exclusive_sparsepart" begin
    Asel = rand_sparsedense(n, r; selector = true)
    @test exclusive_sparsepart(Asel) == Matrix(sparsepart(Asel))[(r + 1):n, :]
    Agen = rand_sparsedense(n, r; selector = false)
    @test_throws ArgumentError exclusive_sparsepart(Agen)
end

@testset "getindex / setindex! round-trips the assembled value" begin
    A = rand_sparsedense(n, r; selector = true)
    # covered row (low-rank correction present): assembled value must read back
    A[1, 1] = 100.0
    @test A[1, 1] ≈ 100.0
    @test sparsepart(A)[1, 1] ≈ 100.0 - fillpart(A)[1, 1]
    # uncovered row: value goes straight into S
    A[5, 5] = 7.0
    @test A[5, 5] ≈ 7.0
    @test sparsepart(A)[5, 5] ≈ 7.0
    # dense U as well
    Ad = rand_sparsedense(n, r; selector = false)
    Ad[2, 4] = -3.0
    @test Ad[2, 4] ≈ -3.0
end

@testset "similar / copy / fill! / Matrix" begin
    A = rand_sparsedense(n, r)
    @test similar(A) isa SparseWithDenseRowColMatrix{Float64}
    @test similar(A, Float32) isa SparseWithDenseRowColMatrix{Float32}
    @test similar(A, Float32, n, n) isa Matrix{Float32}   # dims form → plain Array (FABM parity)

    B = copy(A)
    @test B isa SparseWithDenseRowColMatrix
    @test Matrix(B) ≈ Matrix(A)
    # mutating the copy does not touch the original
    B.V[1, 1] += 1
    @test !(Matrix(B) ≈ Matrix(A))

    C = copy(A)
    fill!(C, 0.0)
    @test iszero(Matrix(C))
    @test nnz(sparsepart(C)) == nnz(sparsepart(A))   # pattern preserved
end

@testset "SelectorMatrix basics" begin
    Sel = SelectorMatrix{Float64}(6, 2)
    @test size(Sel) == (6, 2)
    @test Matrix(Sel) == [Matrix(1.0I, 2, 2); zeros(4, 2)]
    @test_throws ArgumentError SelectorMatrix{Float64}(2, 5)
end
