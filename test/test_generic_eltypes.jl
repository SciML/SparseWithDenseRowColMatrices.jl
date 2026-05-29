include("testutils.jl")
using Test, LinearAlgebra, SparseArrays, ForwardDiff

@testset "Float32 / BigFloat / Complex solve & refactor" begin
    for (T, tol) in (
            (Float32, 1.0f-3), (BigFloat, 1.0e-60), (ComplexF64, 1.0e-9),
            (Complex{BigFloat}, 1.0e-60),
        )
        n, r = 30, 2
        rng = MersenneTwister(1)
        S = sparse(T(4) * I, n, n)
        for i in 1:(n - 1)
            S[i + 1, i] = T(-1)
            S[i, i + 1] = T(-1)
        end
        U = randn(rng, T, n, r)
        V = randn(rng, T, r, n) .* T(0.1)
        A = SparseWithDenseRowColMatrix(S, U, V)
        M = Matrix(A)
        b = randn(rng, T, n)

        F = factorize(A)
        @test eltype(F \ b) == T
        @test relerr(F \ b, M \ b) < tol

        # refactor stays in-type
        S2 = copy(S); S2.nzval .*= T(2)
        A2 = SparseWithDenseRowColMatrix(S2, U, V)
        refactor!(F, S2.nzval)
        @test relerr(F \ b, Matrix(A2) \ b) < tol
    end
end

@testset "ForwardDiff: autodiff through the sparse + low-rank solve" begin
    n, r = 25, 2
    function objective(p)
        T = eltype(p)
        S = sparse(T(4) * p[1] * I, n, n)
        for i in 1:(n - 1)
            S[i + 1, i] = T(-1)
        end
        U = T.(reshape(collect(1.0:(n * r)), n, r)) ./ (n * r) .* p[2]
        V = T.(reshape(collect(1.0:(r * n)), r, n)) ./ (r * n)
        b = T.(collect(1.0:n))
        return sum(factorize(SparseWithDenseRowColMatrix(S, U, V)) \ b)
    end

    p0 = [1.0, 1.0]
    g = ForwardDiff.gradient(objective, p0)
    h = 1.0e-6
    fd = [
        (objective([p0[1] + h, p0[2]]) - objective([p0[1] - h, p0[2]])) / 2h,
        (objective([p0[1], p0[2] + h]) - objective([p0[1], p0[2] - h])) / 2h,
    ]
    @test relerr(g, fd) < 1.0e-5
    @test all(isfinite, g)
end

@testset "ForwardDiff through the augmented fallback" begin
    n, r = 20, 1
    function objective(p)
        T = eltype(p)
        S = sparse(T(2) * I, n, n)
        for i in 1:(n - 1)
            S[i + 1, i] = T(-1)
        end
        for j in 1:n
            S[1, j] = zero(T)
        end     # S singular → augmented path
        Fl = zeros(T, r, n); Fl[1, 1] = p[1]
        A = SparseWithDenseRowColMatrix(S, Fl; replace = true)
        b = T.(collect(1.0:n))
        return sum(factorize(A) \ b)
    end
    g = ForwardDiff.derivative(objective, 1.5)
    h = 1.0e-6
    fd = (objective(1.5 + h) - objective(1.5 - h)) / 2h
    @test isapprox(g, fd; rtol = 1.0e-4)
end
