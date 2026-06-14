include("shared/testutils.jl")
using Test, SparseArrays, LinearAlgebra

@testset "uniformly sparse → not recommended" begin
    # Mimics the user's DAE Jacobians: n≥64, a few nnz per row/col, no dense rows/cols.
    n = 200
    S = sparse(2.0 * I, n, n)
    for i in 1:(n - 1)
        S[i + 1, i] = -1.0; S[i, i + 1] = -1.0
    end
    for _ in 1:(2n)                       # scatter a few extra entries, still ≤ ~8/row
        i = rand(1:n); j = rand(1:n); S[i, j] = 0.3
    end
    rec = recommend_lowrank_peel(S)
    @test rec isa PeelRecommendation
    @test rec.recommended == false
    @test Bool(rec) == false
    @test rec.rank == 0
    @test rec.code === :uniformly_sparse
    @test occursin("uniformly sparse", sprint(show, MIME"text/plain"(), rec))
end

@testset "tridiagonal + dense rows → recommended" begin
    n, k = 2000, 4
    A = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(2.0, n), 1 => fill(-1.0, n - 1))
    A[1:k, :] .= 1.0                      # make the top k rows fully dense
    rec = recommend_lowrank_peel(A)
    @test rec.recommended == true
    @test rec.rank == k
    @test rec.dense_rows == k
    @test rec.code === :recommended
end

@testset "tridiagonal + dense columns → recommended" begin
    n, k = 2000, 3
    A = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(2.0, n), 1 => fill(-1.0, n - 1))
    A[:, 1:k] .= 1.0                      # k dense columns
    rec = recommend_lowrank_peel(A)
    @test rec.recommended == true
    @test rec.rank == k
    @test rec.dense_cols == k
end

@testset "plain tridiagonal → not recommended" begin
    n = 2000
    A = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(2.0, n), 1 => fill(-1.0, n - 1))
    rec = recommend_lowrank_peel(A)
    @test rec.recommended == false
    @test rec.rank == 0
end

@testset "guards" begin
    @test recommend_lowrank_peel(sparse(1.0I, 10, 10)).code === :too_small
    @test recommend_lowrank_peel(sparse(1.0I, 80, 100)).code === :not_square
end

@testset "construction is allocation-light (no eager message string)" begin
    n = 200
    A = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(2.0, n), 1 => fill(-1.0, n - 1))
    recommend_lowrank_peel(A)             # warm up
    a = @allocated recommend_lowrank_peel(A)
    # only the O(n) rowcnt/colcnt scratch + the small struct; the reason text is never built
    # on construction (only by `show`). Generous bound — just a regression guard.
    @test a < 16 * n * sizeof(Int)
    # the descriptive text is available on demand via show
    @test !isempty(sprint(show, MIME"text/plain"(), recommend_lowrank_peel(A)))
end
