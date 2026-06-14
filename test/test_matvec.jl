include("shared/testutils.jl")
using Test, LinearAlgebra

n, r = 40, 3

@testset "A*x and mul! vs dense (dense U)" begin
    A = rand_sparsedense(n, r; selector = false)
    M = Matrix(A)
    x = randn(n)
    @test A * x ≈ M * x

    y = randn(n)
    α, β = 1.7, -0.4
    yref = α .* (M * x) .+ β .* y
    mul!(y, A, x, α, β)
    @test y ≈ yref
end

@testset "A*x and mul! vs dense (selector U)" begin
    A = rand_sparsedense(n, r; selector = true)
    M = Matrix(A)
    x = randn(n)
    @test A * x ≈ M * x
    y = zeros(n)
    mul!(y, A, x)
    @test y ≈ M * x
end

@testset "matrix RHS A*X" begin
    A = rand_sparsedense(n, r)
    M = Matrix(A)
    X = randn(n, 5)
    @test A * X ≈ M * X
    Y = randn(n, 5)
    α, β = 0.5, 2.0
    Yref = α .* (M * X) .+ β .* Y
    mul!(Y, A, X, α, β)
    @test Y ≈ Yref
end

@testset "mul! is allocation-free (vector, selector & dense)" begin
    for sel in (true, false)
        A = rand_sparsedense(n, r; selector = sel)
        x = randn(n)
        y = zeros(n)
        mul!(y, A, x, 1.0, 0.0)            # warm up
        @test (@allocated mul!(y, A, x, 1.0, 0.0)) == 0
    end
end

@testset "SelectorMatrix product" begin
    U = SelectorMatrix{Float64}(7, 3)
    w = [1.0, 2.0, 3.0]
    @test U * w == [1.0, 2.0, 3.0, 0, 0, 0, 0]
    W = reshape(1.0:6.0, 3, 2)
    @test U * W == [Matrix(W); zeros(4, 2)]
end
