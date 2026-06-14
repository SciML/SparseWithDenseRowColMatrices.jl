include("shared/testutils.jl")
using Test, LinearAlgebra, SparseArrays
import IterativeSolvers

# A = S + U V with EXACTLY rank n-1: zero column/row n of S and zero column n of V make column
# n of A identically zero, so A is genuinely rank-deficient while keeping the S+UV structure.
function rankdef(n, r; seed = 1, T = Float64, selector = false)
    rng = MersenneTwister(seed)
    S = spdiagm(0 => fill(T(4), n), 1 => fill(T(-1), n - 1), -1 => fill(T(-1), n - 1))
    S[:, n] .= zero(T); S[n, :] .= zero(T); dropzeros!(S)
    U = selector ? SelectorMatrix{T}(n, r) : T.(randn(rng, n, r))
    V = T.(randn(rng, r, n)); V[:, n] .= zero(T)
    return SparseWithDenseRowColMatrix(S, U, V)
end

minnorm(M, b) = pinv(M) * b      # the oracle: minimum-norm least-squares solution A⁺b

# A = S + U V with S NONSINGULAR and A rank n-k: choose V so the r×r capacitance C = I + V S⁻¹U
# has nullity k (det A = det S · det C). This is the regime the structured-direct engine targets.
function rankdef_nonsingS(n, r, k; seed = 1, T = Float64, selector = false)
    rng = MersenneTwister(seed)
    S = spdiagm(-1 => fill(T(-1), n - 1), 0 => fill(T(4), n), 1 => fill(T(-1), n - 1))
    U = selector ? SelectorMatrix{T}(n, r) : T.(randn(rng, n, r))
    Z = Matrix(S) \ Matrix(U)
    Q = Matrix(qr(T <: Complex ? randn(rng, T, r, r) : randn(rng, r, r)).Q)
    d = ones(T, r)
    for i in 1:k
        d[i] = zero(T)
    end
    V = T.((Q * Diagonal(d) * Q' - I(r)) * pinv(Z))
    return SparseWithDenseRowColMatrix(S, U, V)
end

@testset "adjoint / transpose matvec (core, structured Sᴴu + Vᴴ(Uᴴu))" begin
    for (lbl, A) in (
            ("dense-U", rankdef(70, 3; seed = 1)),
            ("selector-U", rankdef(70, 3; seed = 2, selector = true)),
        )
        M = Matrix(A); u = randn(70)
        @test A' * u ≈ M' * u
        @test transpose(A) * u ≈ transpose(M) * u
        @test mul!(similar(u), A', u) ≈ M' * u
        y = randn(70); ref = 2.0 .* (M' * u) .+ 3.0 .* y   # 5-arg in-place
        mul!(y, A', u, 2.0, 3.0)
        @test y ≈ ref
    end
    # complex: adjoint (conjugating) ≠ transpose
    rng = MersenneTwister(9); n, r = 60, 2
    S = spdiagm(0 => fill(ComplexF64(4), n), 1 => fill(ComplexF64(-1), n - 1), -1 => fill(ComplexF64(-1), n - 1))
    Ac = SparseWithDenseRowColMatrix(S, randn(rng, ComplexF64, n, r), randn(rng, ComplexF64, r, n))
    Mc = Matrix(Ac); u = randn(ComplexF64, n)
    @test Ac' * u ≈ Mc' * u
    @test transpose(Ac) * u ≈ transpose(Mc) * u
    @test !(Ac' * u ≈ transpose(Ac) * u)        # genuinely different for complex
end

@testset "dense engine: minimum-norm LS on rank-deficient A (vs pinv oracle)" begin
    for seed in 1:4
        A = rankdef(80, 3; seed = seed)
        M = Matrix(A)
        @test rank(M) == 79                      # genuinely rank-deficient
        N = nullspace(M)                         # n × 1 here
        for (kind, b) in (
                (:consistent, M * randn(80)),
                (:inconsistent, randn(80)),
            )
            xref = minnorm(M, b)
            x = lstsq(A, b; alg = :dense)
            @test relerr(x, xref) < 1.0e-10
            @test norm(x) ≈ norm(xref) rtol = 1.0e-9          # minimum-norm
            @test norm(M' * (M * x - b)) < 1.0e-8 * norm(M)   # least-squares optimality
            @test norm(N' * x) < 1.0e-9                       # no null-space component
            kind === :consistent && @test norm(M * x - b) < 1.0e-9
        end
    end
end

@testset "structured direct engine: min-norm LS without densifying (S nonsingular)" begin
    for seed in 1:3, (n, r, k) in ((60, 3, 1), (90, 5, 2), (120, 8, 3)), sel in (false, true)
        A = rankdef_nonsingS(n, r, k; seed = seed, selector = sel)
        M = Matrix(A)
        @test rank(M) == n - k
        N = nullspace(M)
        for (kind, b) in ((:consistent, M * randn(n)), (:inconsistent, randn(n)))
            xref = minnorm(M, b)
            x = lstsq(A, b; alg = :structured)
            @test relerr(x, xref) < 1.0e-9
            @test norm(x) ≈ norm(xref) rtol = 1.0e-8          # minimum-norm
            @test norm(M' * (M * x - b)) < 1.0e-7 * norm(M)   # least-squares optimality
            @test norm(N' * x) < 1.0e-8                       # no null-space component
            kind === :consistent && @test norm(M * x - b) < 1.0e-8
        end
    end
end

@testset "structured engine: complex" begin
    n, r, k = 70, 3, 1
    rng = MersenneTwister(5)
    S = spdiagm(0 => fill(ComplexF64(4), n), -1 => fill(ComplexF64(-1), n - 1), 1 => fill(ComplexF64(-1), n - 1))
    U = randn(rng, ComplexF64, n, r); Z = Matrix(S) \ U
    Q = Matrix(qr(randn(rng, ComplexF64, r, r)).Q); d = ones(r); d[1] = 0
    V = ComplexF64.((Q * Diagonal(d) * Q' - I(r)) * pinv(Z))
    A = SparseWithDenseRowColMatrix(S, U, V); M = Matrix(A)
    b = randn(ComplexF64, n)
    @test relerr(lstsq(A, b; alg = :structured), minnorm(M, b)) < 1.0e-9
end

@testset "structured engine peels a selector-singular S (BVP boundary case)" begin
    # S has its boundary rows zeroed (singular); U is a SelectorMatrix supplying them. The sparse
    # peel S̃ = S + U Uᴴ (r diagonal entries) keeps the structured DIRECT path — no dense fallback —
    # and stays exact, for A nonsingular AND rank-deficient, consistent AND inconsistent.
    for seed in 1:3, rde in (false, true)
        n, r = 100, 3
        S = spdiagm(0 => fill(3.0, n), 1 => fill(-1.0, n - 1), -1 => fill(-1.0, n - 1))
        S[1:r, :] .= 0.0; dropzeros!(S)
        F = zeros(r, n)
        for i in 1:r
            F[i, i] = 1.0; F[i, i + r] = 0.3
        end
        rde && (F[r, :] .= F[1, :])                  # make A itself rank-deficient
        A = SparseWithDenseRowColMatrix(S, F; replace = true)
        M = Matrix(A); N = nullspace(M)
        @test rank(M) == n - (rde ? 1 : 0)
        for (kind, b) in ((:cons, M * randn(n)), (:incons, randn(n)))
            xref = minnorm(M, b)
            x = lstsq(A, b; alg = :structured)       # must NOT throw — the peel succeeds
            @test relerr(x, xref) < 1.0e-9
            @test norm(x) ≈ norm(xref) rtol = 1.0e-8
            isempty(N) || @test norm(N' * x) < 1.0e-8
        end
    end
end

@testset ":auto uses structured on nonsingular S, falls back on singular/near-singular S" begin
    # nonsingular S: :auto path agrees with explicit :structured and :dense, all = pinv
    A = rankdef_nonsingS(100, 4, 2; seed = 1)
    M = Matrix(A); b = randn(100); xref = minnorm(M, b)
    @test relerr(lstsq(A, b), xref) < 1.0e-9                       # :auto default
    @test relerr(lstsq(A, b; alg = :auto), xref) < 1.0e-9
    @test lstsq(A, b; alg = :auto) ≈ lstsq(A, b; alg = :structured)
    @test lstsq(A, b; alg = :auto) ≈ lstsq(A, b; alg = :dense)

    # exactly-singular S (zeroed row): structured klu throws → :auto falls back to dense
    Asing = rankdef(80, 3; seed = 2)
    Msing = Matrix(Asing); bs = randn(80)
    @test relerr(lstsq(Asing, bs; alg = :auto), minnorm(Msing, bs)) < 1.0e-9
    @test_throws ArgumentError lstsq(Asing, bs; alg = :structured)

    # near-singular S (κ(S) ≈ 5e13, but A well-conditioned): structured would be silently wrong;
    # :auto must fall back to the exact dense COD and :structured must error (not return garbage)
    n = 80; rng = MersenneTwister(3)
    Snear = spdiagm(0 => collect(exp.(range(0, -log(4.7e13); length = n))), -1 => fill(-1.0e-3, n - 1))
    Anear = SparseWithDenseRowColMatrix(Snear, randn(rng, n, 2), randn(rng, 2, n))
    Mnear = Matrix(Anear); bn = randn(n)
    @test cond(Mnear) < 1.0e8                                      # A itself is well-conditioned
    @test relerr(lstsq(Anear, bn; alg = :auto), minnorm(Mnear, bn)) < 1.0e-7
    @test_throws ArgumentError lstsq(Anear, bn; alg = :structured)
end

@testset "structured rejects Tikhonov λ" begin
    A = rankdef_nonsingS(40, 2, 1; seed = 1)
    @test_throws ArgumentError lstsq(A, randn(40); alg = :structured, λ = 0.1)
end

@testset "iterative engine recovers the same minimum-norm solution" begin
    for seed in 1:3, solver in (:lsqr, :lsmr)
        A = rankdef(80, 3; seed = seed)
        M = Matrix(A)
        b = seed == 1 ? M * randn(80) : randn(80)
        xref = minnorm(M, b)
        x = lstsq(A, b; alg = :iterative, solver = solver, atol = 1.0e-13, btol = 1.0e-13)
        @test relerr(x, xref) < 1.0e-6
        @test norm(x) ≈ norm(xref) rtol = 1.0e-5
        @test relerr(x, lstsq(A, b)) < 1.0e-6     # agrees with the dense engine
    end
end

@testset "SelectorMatrix-U (boundary-condition form)" begin
    A = rankdef(80, 2; seed = 5, selector = true)
    M = Matrix(A); b = randn(80)
    xref = minnorm(M, b)
    @test relerr(lstsq(A, b), xref) < 1.0e-10
    @test relerr(lstsq(A, b; alg = :iterative), xref) < 1.0e-6
end

@testset "full-rank square reduces to the ordinary solve" begin
    A = rand_sparsedense(60, 3; seed = 11)
    M = Matrix(A); b = randn(60)
    @test relerr(lstsq(A, b), M \ b) < 1.0e-10
    @test relerr(lstsq(A, b; alg = :iterative), M \ b) < 1.0e-6
end

@testset "complex rank-deficient LS" begin
    rng = MersenneTwister(21); n, r = 70, 2
    S = spdiagm(0 => fill(ComplexF64(4), n), -1 => fill(ComplexF64(-1), n - 1))
    S[:, n] .= 0; S[n, :] .= 0; dropzeros!(S)
    V = randn(rng, ComplexF64, r, n); V[:, n] .= 0
    A = SparseWithDenseRowColMatrix(S, randn(rng, ComplexF64, n, r), V)
    M = Matrix(A); b = randn(ComplexF64, n)
    @test rank(M) == n - 1
    @test relerr(lstsq(A, b), minnorm(M, b)) < 1.0e-10
    @test relerr(lstsq(A, b; alg = :iterative), minnorm(M, b)) < 1.0e-6
end

@testset "Tikhonov damping λ > 0 (both engines)" begin
    A = rankdef(60, 2; seed = 7)
    M = Matrix(A); b = randn(60); λ = 0.1
    xref = (M' * M + λ^2 * I) \ (M' * b)         # the damped normal-equations solution
    @test relerr(lstsq(A, b; λ = λ), xref) < 1.0e-9
    @test relerr(lstsq(A, b; alg = :iterative, λ = λ, atol = 1.0e-13, btol = 1.0e-13), xref) < 1.0e-6
end

@testset "cached factorization (SparseWithDenseRowColLeastSquares) reuses S for repeated solves" begin
    for sel in (false, true)
        A = rankdef_nonsingS(100, 4, 2; seed = 1, selector = sel)
        M = Matrix(A)
        F = SparseWithDenseRowColLeastSquares(A)
        @test denserank(F) == 4
        @test size(F) == (100, 100)
        for b in (M * randn(100), randn(100))         # consistent, inconsistent
            xref = minnorm(M, b)
            @test relerr(F \ b, xref) < 1.0e-9
            x = copy(b); ldiv!(F, x)                   # in-place (alias-safe)
            @test relerr(x, xref) < 1.0e-9
            y = similar(b); ldiv!(y, F, b)             # 3-arg
            @test relerr(y, xref) < 1.0e-9
        end
    end
    # the cached factorization agrees with the one-shot lstsq on the SAME rhs
    A = rankdef_nonsingS(80, 3, 1; seed = 2)
    b = randn(80)
    @test SparseWithDenseRowColLeastSquares(A) \ b ≈ lstsq(A, b; alg = :structured)
    # selector-singular S (BVP): the cached factorization peels and works
    n, r = 80, 3
    S = spdiagm(0 => fill(3.0, n), 1 => fill(-1.0, n - 1), -1 => fill(-1.0, n - 1))
    S[1:r, :] .= 0.0; dropzeros!(S)
    Fb = zeros(r, n); for i in 1:r
        Fb[i, i] = 1.0; Fb[i, i + r] = 0.3
    end
    Abvp = SparseWithDenseRowColMatrix(S, Fb; replace = true); Mbvp = Matrix(Abvp); bb = randn(n)
    @test relerr(SparseWithDenseRowColLeastSquares(Abvp) \ bb, minnorm(Mbvp, bb)) < 1.0e-9
    # general singular S: the constructor errors (points the user to alg=:dense)
    @test_throws ArgumentError SparseWithDenseRowColLeastSquares(rankdef(40, 2; seed = 3))
end

@testset "lstsq is a separate opt-in: \\ / factorize / qr still throw on singular A" begin
    A = rankdef(50, 2; seed = 3)
    b = randn(50)
    @test_throws SingularException factorize(A; strategy = :augmented)
    @test_throws SingularException qr(A)
    @test_throws SingularException (A \ b)
    @test relerr(lstsq(A, b), minnorm(Matrix(A), b)) < 1.0e-10   # but lstsq succeeds
end

@testset "eltype guard and argument validation" begin
    A = rankdef(20, 2; seed = 1, T = BigFloat)   # gelsy / BLAS-iterative cannot do BigFloat
    @test_throws ArgumentError lstsq(A, randn(BigFloat, 20))
    # non-BLAS-float (Int) must be rejected, not silently promoted to a Float64 answer
    Si = spdiagm(0 => fill(4, 20), 1 => fill(-1, 19), -1 => fill(-1, 19))
    Ui = zeros(Int, 20, 2); Ui[1, 1] = 1; Ui[2, 2] = 1
    Vi = zeros(Int, 2, 20); Vi[1, 1] = 1; Vi[2, 2] = 1
    Ai = SparseWithDenseRowColMatrix(Si, Ui, Vi)
    @test_throws ArgumentError lstsq(Ai, ones(Int, 20))
    @test_throws ArgumentError lstsq(Ai, ones(Int, 20); alg = :iterative)

    Af = rankdef(20, 2; seed = 1)
    bf = randn(20)
    @test_throws ArgumentError lstsq(Af, bf; alg = :bogus)
    @test_throws ArgumentError lstsq(Af, bf; alg = :iterative, solver = :bogus)
    @test_throws DimensionMismatch lstsq(Af, randn(19))
    # negative Tikhonov damping rejected by BOTH engines (dense must not silently use |λ|)
    @test_throws ArgumentError lstsq(Af, bf; λ = -1.0)
    @test_throws ArgumentError lstsq(Af, bf; alg = :iterative, λ = -1.0)
end

@testset "iterative non-convergence warns (isconverged is unreliable for lsqr)" begin
    A = rankdef(60, 3; seed = 4)
    b = randn(60)
    # a far-too-small budget must WARN for both solvers (the bug was lsqr silently returning a
    # wrong answer with isconverged=true; we now flag on budget exhaustion)
    @test_logs (:warn,) match_mode = :any lstsq(A, b; alg = :iterative, solver = :lsqr, maxiters = 2)
    @test_logs (:warn,) match_mode = :any lstsq(A, b; alg = :iterative, solver = :lsmr, maxiters = 2)
    # an ample budget converges, so the result is accurate (and the warn branch is not taken)
    @test relerr(lstsq(A, b; alg = :iterative, maxiters = 20 * 60), minnorm(Matrix(A), b)) < 1.0e-6
end
