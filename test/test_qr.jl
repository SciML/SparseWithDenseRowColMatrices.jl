include("testutils.jl")
using Test, LinearAlgebra, SparseArrays

# A = S + U V with an ILL-conditioned sparse part S (κ(S) ≈ `kappa`) but a well-conditioned
# assembled A (κ(A) ≈ 6): the top `r` rows of the tridiagonal S are scaled by 1/kappa (making
# S nearly rank-deficient there) and the dense fill replaces exactly those rows with O(1)
# entries. This is the FastAlmostBandedMatrices regime where forming S⁻¹ (Woodbury) loses
# accuracy but the augmented QR — which never forms S⁻¹ — does not.
function illcond_S(n, r, kappa; T = Float64)
    S = spdiagm(0 => fill(T(4), n), 1 => fill(T(-1), n - 1), -1 => fill(T(-1), n - 1))
    for i in 1:r
        S[i, :] .*= inv(T(kappa))
    end
    dropzeros!(S)
    F = zeros(T, r, n)
    for i in 1:r
        F[i, i] = one(T)
        F[i, i + r] = T(0.3)
    end
    return SparseWithDenseRowColMatrix(S, F; replace = true)
end

# Exact solve of the (rounded) working-precision matrix, computed in BigFloat — the reference
# every forward-error assertion compares against (matches how the design was measured).
function exact_solve(A, b)
    x = big.(Matrix(A)) \ big.(b)
    return convert.(eltype(b), x)
end


@testset "well-conditioned: qr agrees with factorize at the noise floor" begin
    A = rand_sparsedense(300, 4; seed = 7)
    b = randn(300)
    xref = exact_solve(A, b)
    F = qr(A)
    @test F isa SparseWithDenseRowColQRAugmented
    @test issuccess(F)
    @test size(F) == (300, 300)
    @test denserank(F) == 4
    @test relerr(F \ b, xref) < 1.0e-12
    @test relerr(factorize(A) \ b, xref) < 1.0e-10   # both at noise floor
end

@testset "ill-conditioned S: augmented QR holds where Woodbury degrades" begin
    n, r = 200, 4
    b = randn(MersenneTwister(0), n)
    # The load-bearing regression guard: across the entire κ(S) sweep — including κ(S) ≥ 3.6e11
    # where Woodbury (even with refinement) loses accuracy — the augmented QR stays ≤ 1e-13.
    for kappa in (3.6e6, 3.6e10, 3.6e11, 3.6e12, 3.6e14)
        A = illcond_S(n, r, kappa)
        xref = exact_solve(A, b)
        F = qr(A)
        @test issuccess(F)
        @test relerr(F \ b, xref) < 1.0e-13
    end
    # And at the top of the sweep the QR is decisively better than the default Woodbury path.
    A = illcond_S(n, r, 3.6e14)
    xref = exact_solve(A, b)
    err_qr = relerr(qr(A) \ b, xref)
    err_wood = relerr(factorize(A; strategy = :woodbury, auto_fallback = false, refine = 2) \ b, xref)
    @test err_qr < 1.0e-13
    @test err_wood > 1.0e3 * err_qr
end

@testset "strategy handling and no Woodbury-QR footgun" begin
    A = rand_sparsedense(40, 2; seed = 2)
    @test qr(A; strategy = :auto) isa SparseWithDenseRowColQRAugmented
    @test qr(A; strategy = :augmented) isa SparseWithDenseRowColQRAugmented
    @test_throws ArgumentError qr(A; strategy = :woodbury)
    @test_throws ArgumentError qr(A; strategy = :bogus)
    # The Woodbury-over-qr(S) mode is intentionally not shipped (shares the κ(S)·κ(C)
    # cancellation): there must be no such type to silently re-introduce it.
    @test !isdefined(SparseWithDenseRowColMatrices, :SparseWithDenseRowColQR)
end

@testset "eltypes: BLAS floats, complex, and BigFloat all work" begin
    # The column-pivoted backend is pure Julia, so the QR path is no longer BLAS-float-only:
    # Float32/ComplexF32 work on every Julia version, and generic floats (BigFloat) work too.
    for (T, tol) in (
            (Float32, 1.0e-4), (Float64, 1.0e-11), (ComplexF32, 1.0e-4),
            (ComplexF64, 1.0e-11), (BigFloat, 1.0e-30),
        )
        A = rand_sparsedense(50, 3; seed = 4, T = T)
        b = T <: Complex ? (randn(50) .+ im .* randn(50)) .|> T : T.(randn(50))
        F = qr(A)
        @test F isa SparseWithDenseRowColQRAugmented{T}
        @test relerr(F \ b, exact_solve(A, b)) < tol
    end
    # Non-float eltypes that cannot support a QR (no sqrt/division) still error.
    nn = 20
    Si = spdiagm(0 => fill(4, nn), 1 => fill(-1, nn - 1), -1 => fill(-1, nn - 1))
    Ui = zeros(Int, nn, 2); Ui[1, 1] = 1; Ui[2, 2] = 1
    Vi = zeros(Int, 2, nn); Vi[1, 1] = 1; Vi[2, 2] = 1
    @test_throws Exception qr(SparseWithDenseRowColMatrix(Si, Ui, Vi))
end

@testset "adjoint & transpose solve (lazy cached QaugH)" begin
    A = rand_sparsedense(120, 3; seed = 11)
    M = Matrix(A)
    b = randn(120)
    F = qr(A)
    @test F.QaugH === nothing                      # not built until first adjoint/transpose solve
    @test relerr(F' \ b, M' \ b) < 1.0e-11
    @test F.QaugH !== nothing                    # lazily built (a CSRQRFactorization)
    cached = F.QaugH
    @test relerr(F' \ b, M' \ b) < 1.0e-11
    @test F.QaugH === cached                       # second adjoint solve reuses it (no rebuild)
    @test relerr(transpose(F) \ b, transpose(M) \ b) < 1.0e-11
    x = copy(b)
    ldiv!(F', x)
    @test relerr(x, M' \ b) < 1.0e-11
    # complex A: adjoint ≠ transpose
    Ac = rand_sparsedense(80, 2; seed = 12, T = ComplexF64)
    Mc = Matrix(Ac)
    bc = randn(ComplexF64, 80)
    Fc = qr(Ac)
    @test relerr(Fc' \ bc, Mc' \ bc) < 1.0e-10
    @test relerr(transpose(Fc) \ bc, transpose(Mc) \ bc) < 1.0e-10
end

@testset "complex RHS over a real factorization" begin
    A = rand_sparsedense(60, 2; seed = 13)
    M = Matrix(A)
    b = randn(60) .+ im .* randn(60)
    F = qr(A)
    @test relerr(F \ b, M \ b) < 1.0e-11
    @test relerr(F' \ b, M' \ b) < 1.0e-11
end

@testset "matrix RHS and 3-arg ldiv!" begin
    A = rand_sparsedense(90, 3; seed = 14)
    M = Matrix(A)
    B = randn(90, 5)
    F = qr(A)
    @test relerr(F \ B, M \ B) < 1.0e-11
    Y = similar(B)
    ldiv!(Y, F, B)
    @test relerr(Y, M \ B) < 1.0e-11
end

@testset "refactor! / qr! (reuses symbolic analysis)" begin
    A1 = rand_sparsedense(100, 3; seed = 21)
    A2 = rand_sparsedense(100, 3; seed = 22)   # same pattern, new values
    b = randn(100)
    F = qr(A1)
    _ = F' \ b                                 # force QaugH to be built
    @test F.QaugH !== nothing                    # lazily built (a CSRQRFactorization)
    refactor!(F, A2)
    @test F.QaugH === nothing                  # adjoint cache invalidated on refactor
    @test relerr(F \ b, Matrix(A2) \ b) < 1.0e-11
    # qr! is an alias for refactor!
    A3 = rand_sparsedense(100, 3; seed = 23)
    qr!(F, A3)
    @test relerr(F \ b, Matrix(A3) \ b) < 1.0e-11
    # a shape change must be rejected
    Abig = rand_sparsedense(101, 3; seed = 24)
    @test_throws DimensionMismatch refactor!(F, Abig)
    Arank = rand_sparsedense(100, 4; seed = 25)
    @test_throws DimensionMismatch refactor!(F, Arank)
end

@testset "singularity: singular A throws, ill-conditioned S succeeds" begin
    # A genuinely singular A (a zero column the dense fill does not supply) ⇒ the bordered
    # system is rank-deficient ⇒ SingularException, matching the LU path's contract.
    n, r = 40, 2
    S = spdiagm(0 => fill(3.0, n), -1 => fill(-1.0, n - 1))
    S[:, n] .= 0.0
    S[n, :] .= 0.0
    dropzeros!(S)
    F = zeros(r, n)
    for i in 1:r
        F[i, i] = 1.0          # dense rows do not touch column n ⇒ A column n is all zero
    end
    Asing = SparseWithDenseRowColMatrix(S, F; replace = true)
    @test_throws SingularException qr(Asing)
    @test_throws SingularException factorize(Asing; strategy = :augmented)  # LU agrees

    # Ill-conditioned S but nonsingular A must SUCCEED (the whole point of the path).
    Aok = illcond_S(80, 3, 3.6e13)
    Fok = qr(Aok)
    @test issuccess(Fok)
    b = randn(80)
    @test relerr(Fok \ b, exact_solve(Aok, b)) < 1.0e-13
end

@testset "ill-conditioned ASSEMBLED A does not spuriously throw" begin
    # Regression: a rank-revealing QR must not flag an ill-conditioned-but-nonsingular bordered
    # system as rank-deficient. An assembled A up to cond ~1e13 must factor and solve to roughly
    # its conditioning-limited accuracy (κ(A)·eps), matching the LU path's behavior.
    function illcond_assembled(n; kappa)
        d = collect(exp.(range(0, -log(kappa); length = n)))    # graded 1 .. 1/kappa
        S = spdiagm(0 => d)
        U = zeros(n, 1); U[1, 1] = 1.0e-8
        V = zeros(1, n); V[1, 1] = 1.0e-8                        # negligible low-rank part
        return SparseWithDenseRowColMatrix(S, U, V)
    end
    b = randn(MersenneTwister(0), 120)
    for kappa in (1.0e10, 1.0e12, 1.0e13)
        A = illcond_assembled(120; kappa = kappa)
        F = qr(A)                                                # must NOT throw
        @test issuccess(F)
        xref = Float64.(big.(Matrix(A)) \ big.(b))
        # backward-stable: forward error tracks κ(A)·eps, not the noise floor
        @test relerr(F \ b, xref) < 1.0e3 * cond(Matrix(A)) * eps()
        # the augmented-LU path also solves these (contract parity — neither throws)
        @test issuccess(factorize(A; strategy = :augmented))
    end
end

@testset "update_lowrank! is not supported on the QR path" begin
    A = rand_sparsedense(30, 2; seed = 31)
    F = qr(A)
    @test_throws ArgumentError update_lowrank!(F)
end

@testset "solve is allocation-free" begin
    A = rand_sparsedense(200, 4; seed = 41)
    b = randn(200)
    F = qr(A)
    x = copy(b); ldiv!(F, x)                      # warm
    y = copy(b)
    @test (@allocated ldiv!(F, y)) == 0           # in-place solve allocates nothing
end

@testset "r == 0 edge case reduces to qr(S)" begin
    n = 50
    S = spdiagm(0 => fill(4.0, n), 1 => fill(-1.0, n - 1), -1 => fill(-1.0, n - 1))
    A = SparseWithDenseRowColMatrix(S, zeros(0, n))   # no dense rows/cols
    @test denserank(A) == 0
    F = qr(A)
    @test denserank(F) == 0
    @test issuccess(F)
    b = randn(n)
    @test relerr(F \ b, Matrix(S) \ b) < 1.0e-11
    @test relerr(F' \ b, Matrix(S)' \ b) < 1.0e-11
end
