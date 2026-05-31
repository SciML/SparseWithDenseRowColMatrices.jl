# ------------------
# Rank-deficient / inconsistent least squares: minimum-norm solution x = A⁺b
# ------------------
#
# `\`, `factorize`/`lu`, and `qr` solve NONSINGULAR systems and throw on a singular `A`. For a
# genuinely rank-deficient or inconsistent system one wants the Moore–Penrose solution
# x = A⁺b (the minimum-2-norm vector among the minimizers of ‖Ax−b‖). The bordered/Woodbury
# machinery CANNOT produce it — solving `[S U; V -I][x;y]=[b;0]` in least squares minimizes a
# different objective than `‖Ax−b‖`, and there is no Woodbury-style formula for the
# pseudoinverse of a sum. So this is a separate, opt-in entry point.
#
# Engines:
#   :auto (default)      — try :structured, fall back to :dense when S is singular/near-singular.
#   :structured          — DIRECT and structure-exploiting (no densifying). When S is nonsingular,
#                          A = S(I + ZV) with Z = S⁻¹U, and the entire rank deficiency collapses
#                          into the small r×r capacitance C = I + V Z (nullity(A) = nullity(C)).
#                          So A⁺b is built from ONE sparse factorization of S (PureKLU) + r solves
#                          (Z) + small dense SVD/QR on r×r and n×s (s ≤ 2r) blocks — never forming
#                          or factoring the dense A. O(factor(S) + n·r²); >100× faster than dense
#                          COD for large n. Requires S nonsingular & well-conditioned (guarded).
#   :dense               — complete-orthogonal decomposition (LAPACK gelsy) of the densified A.
#                          Exact (matches `pinv(Matrix(A))*b`), O(n³) time / O(n²) memory; the
#                          mandatory fallback when S is singular/near-singular (where the
#                          structured route is silently wrong).
#   :iterative           — LSQR/LSMR (IterativeSolvers) driven by A's structured matvec/adjoint,
#                          never forming the dense A. For n too large to densify when S is also
#                          singular (so neither :structured nor :dense apply). Approximate to a
#                          tolerance, fragile under ill-conditioning. Lives in an extension.

_lstsq_supported(::Type{<:Union{Float32, Float64, ComplexF32, ComplexF64}}) = true
_lstsq_supported(::Type) = false

function _check_lstsq_eltype(::Type{T}) where {T}
    _lstsq_supported(T) || throw(
        ArgumentError(
            "lstsq supports only Float32/Float64/ComplexF32/ComplexF64 (it uses LAPACK / a BLAS-float " *
                "iterative solver); got eltype $T."
        )
    )
    return nothing
end

# Iterative engine — method added by the IterativeSolvers extension; no method in core.
function _lstsq_iterative end

"""
    lstsq(A::SparseWithDenseRowColMatrix, b::AbstractVector; alg=:auto, kwargs...) -> x

Minimum-norm least-squares solution `x ≈ A⁺b` (the Moore–Penrose pseudoinverse solution),
correct for **rank-deficient and/or inconsistent** systems where `\\`/`factorize`/`qr` would
throw. Among all `x` minimizing `‖A x − b‖₂`, returns the one of smallest `‖x‖₂`. Both the
consistent and inconsistent cases are handled (no user flag).

* `alg = :auto` (default): use the structure-exploiting **direct** method when `S` is
  nonsingular and well-conditioned, automatically falling back to `:dense` otherwise.
* `alg = :structured`: a **direct** solve that never densifies `A`. When `S` is nonsingular the
  rank deficiency collapses into the small `r × r` capacitance `C = I + V S⁻¹U`, so `A⁺b` comes
  from one sparse factorization of `S` (PureKLU) plus small dense work on `r × r` / `n × s`
  (`s ≤ 2r`) blocks — `O(factor(S) + n·r²)`, `>100×` faster than `:dense` for large `n`. Errors
  if `S` is singular/near-singular (where the structured route would be silently wrong).
* `alg = :dense`: complete-orthogonal decomposition (LAPACK `gelsy`) of the densified `A`.
  Exact, `O(n³)` / `O(n²)`; the mandatory fallback when `S` is singular.
* `alg = :iterative`: LSQR / LSMR via `A`'s structured matvec/adjoint, never densifying — for
  `n` too large to densify *and* `S` singular. Requires `using IterativeSolvers`; approximate.

Keywords: `tolC` (`:structured`/`:auto` rank cutoff for `C`), `rcond` (`:dense` rank cutoff),
`solver` (`:lsqr`/`:lsmr`), `atol`, `btol`, `maxiters`, `λ` (Tikhonov damping; `0` = pure
minimum-norm), `verbose`. Only `Float32`/`Float64`/`ComplexF32`/`ComplexF64` are supported.

`\\`, `factorize`/`lu`, and `qr` are unchanged — they stay exact and throw on singular `A`;
least squares is always an explicit opt-in via `lstsq`.
"""
function lstsq(A::SparseWithDenseRowColMatrix, b::AbstractVector; alg::Symbol = :auto, kwargs...)
    if alg === :auto
        # structured can't do Tikhonov; route λ>0 (and structured-unsafe S) to the exact dense COD
        if iszero(get(kwargs, :λ, 0))
            x, ok = _lstsq_structured(A, b; kwargs...)
            ok && return x
        end
        return _lstsq_dense(A, b; kwargs...)
    elseif alg === :structured
        x, ok = _lstsq_structured(A, b; kwargs...)
        ok || throw(
            ArgumentError(
                "lstsq(...; alg=:structured) is unsafe for this A: S is singular or near-singular, " *
                    "so the structured route (which inverts S) would be silently wrong. Use alg=:dense " *
                    "(exact) or alg=:auto (auto-fallback)."
            )
        )
        return x
    elseif alg === :dense
        return _lstsq_dense(A, b; kwargs...)
    elseif alg === :iterative
        isempty(methods(_lstsq_iterative)) && throw(
            ArgumentError(
                "lstsq(...; alg=:iterative) requires `using IterativeSolvers` (the iterative engine " *
                    "lives in a package extension)."
            )
        )
        return _lstsq_iterative(A, b; kwargs...)
    else
        throw(ArgumentError("alg must be :auto, :structured, :dense or :iterative; got :$alg"))
    end
end

# Structure-exploiting DIRECT min-norm least-squares (returns `(x, ok)`; `ok=false` signals the
# caller to fall back to dense because S is singular/near-singular). With S nonsingular,
# A = S(I + ZV), Z = S⁻¹U; the rank deficiency lives in C = I + V Z (nullity(A)=nullity(C)).
# A⁺b is assembled from one PureKLU factorization of S + small dense SVD/QR — A is never formed.
function _lstsq_structured(
        A::SparseWithDenseRowColMatrix, b::AbstractVector;
        tolC::Real = -one(real(float(eltype(A)))), λ::Real = 0,
        # accept and ignore the other engines' keywords so `alg` can be flipped without edits
        rcond = nothing, solver = nothing, atol = nothing, btol = nothing,
        maxiters = nothing, verbose = nothing,
    )
    n = size(A, 1)
    r = denserank(A)
    length(b) == n || throw(DimensionMismatch("A has $n columns, b has length $(length(b))"))
    λ ≥ 0 || throw(ArgumentError("λ (Tikhonov damping) must be ≥ 0; got $λ"))
    iszero(λ) || throw(ArgumentError("lstsq(...; alg=:structured) does not support Tikhonov λ>0; use alg=:dense."))
    RT = promote_type(eltype(A), eltype(b))
    _check_lstsq_eltype(RT)
    TT = float(RT)
    rT = real(TT)
    tol = tolC < 0 ? sqrt(eps(rT)) : rT(tolC)

    Sown = _own_sparse(TT, A.S)
    Sfact = try
        PureKLU.klu(Sown)
    catch e
        e isa SingularException || rethrow(e)
        return (zeros(TT, n), false)                # S exactly singular → fall back to dense
    end

    if r == 0                                       # A = S nonsingular ⇒ the solution is S⁻¹b
        x = collect(TT, b)
        PureKLU.solve!(Sfact, x)
        return (x, true)
    end

    V = Matrix{TT}(A.V)                             # r × n
    Z = Matrix{TT}(undef, n, r)
    materialize_U!(Z, A.U)                          # Z = U
    nrmU = norm(Z)
    PureKLU.solve!(Sfact, Z)                        # Z = S⁻¹U

    # Guard: near-singular S makes Z = S⁻¹U huge/garbage and `klu` does NOT throw, which both
    # crashes the SVDs and gives a silently-wrong answer. κ̂(S) = ‖S‖₁·‖Z‖/‖U‖ ≳ κ(S); bail early.
    kappaS = nrmU > 0 ? opnorm(Sown, 1) * norm(Z) / nrmU : zero(rT)
    (all(isfinite, Z) && isfinite(kappaS) && kappaS ≤ inv(sqrt(eps(rT)))) ||
        return (zeros(TT, n), false)

    Vadj = Matrix(V')                              # n × r  (adjoint; conjugate for complex)
    C = V * Z                                       # r × r capacitance
    @inbounds for i in 1:r
        C[i, i] += one(TT)
    end
    Fc = svd(C)
    kdef = count(<(tol * Fc.S[1]), Fc.S)            # nullity(A) = nullity(C)

    # orthonormal basis W (n × s, s ≤ 2r) of the active subspace span([Z | Vᴴ]); M is the
    # identity off this subspace, so Mᴴ-pseudoinverse work happens only on the small s × s block.
    QB = qr(hcat(Z, Vadj), ColumnNorm())
    Rd = abs.(diag(QB.R))
    s = isempty(Rd) ? 0 : count(>(tol * maximum(Rd)), Rd)
    W = Matrix(QB.Q)[:, 1:s]
    Msmall = W' * (Z * (V * W))                     # s × s
    @inbounds for i in 1:s
        Msmall[i, i] += one(TT)
    end
    Fs = svd(Msmall)
    sp = [σ > tol * Fs.S[1] ? inv(σ) : zero(rT) for σ in Fs.S]   # truncated pinv of Msmall

    btil = collect(TT, b)
    if kdef > 0                                     # project b onto range(A) (no-op when full rank)
        Nl = Fc.U[:, (end - kdef + 1):end]          # null(Cᴴ)  (r × kdef)
        LrA = Vadj * Nl                             # n × kdef
        PureKLU.solve!(Sfact', LrA)                 # left-null(A) = S⁻ᴴ Vᴴ null(Cᴴ)
        QL = Matrix(qr(LrA).Q)[:, 1:kdef]
        btil .-= QL * (QL' * btil)
    end
    PureKLU.solve!(Sfact, btil)                     # btil = S⁻¹ b̃  (= c)
    Wtc = W' * btil
    x = W * (Fs.V * (sp .* (Fs.U' * Wtc))) + (btil - W * Wtc)   # x = Mᴴ⁺ c

    # post-hoc least-squares optimality check (cheap structured matvecs): catches a misjudged
    # rank/nullity. ‖Aᴴ(Ax−b)‖ → 0 at a minimizer; a loose threshold separates a good solution
    # (~1e-14 relative) from a grossly wrong one (~1). A false trip only falls `:auto` back to the
    # exact dense path, so erring conservative is safe.
    res = A * x
    res .-= b
    atr = A' * res
    nrm_atr = norm(atr)
    ok = nrm_atr ≤ 1.0e-6 * (norm(A' * collect(TT, b)) + nrm_atr)
    return (x, ok)
end

function _lstsq_dense(
        A::SparseWithDenseRowColMatrix, b::AbstractVector;
        rcond::Real = -one(real(float(eltype(A)))), λ::Real = 0,
        # accept and ignore iterative-only keywords so `alg` can be flipped without edits
        solver = nothing, atol = nothing, btol = nothing, maxiters = nothing, verbose = nothing,
    )
    n = size(A, 1)
    length(b) == n || throw(DimensionMismatch("A has $n columns, b has length $(length(b))"))
    λ ≥ 0 || throw(ArgumentError("λ (Tikhonov damping) must be ≥ 0; got $λ"))
    _check_lstsq_eltype(promote_type(eltype(A), eltype(b)))   # reject non-BLAS-float (e.g. Int) up front
    TT = float(promote_type(eltype(A), eltype(b)))
    rc = rcond < 0 ? eps(real(TT)) * n : real(TT)(rcond)
    M = Matrix(A)
    Md = eltype(M) === TT ? copy(M) : TT.(M)
    if iszero(λ)
        rhs = collect(TT, b)
        x, _ = LinearAlgebra.LAPACK.gelsy!(Md, rhs, rc)
        return x
    else
        # Tikhonov: minimize ‖A x − b‖² + λ²‖x‖² = ‖[A; λI] x − [b; 0]‖²
        stacked = vcat(Md, real(TT)(λ) * Matrix{TT}(LinearAlgebra.I, n, n))
        rhs = vcat(collect(TT, b), zeros(TT, n))
        x, _ = LinearAlgebra.LAPACK.gelsy!(stacked, rhs, rc)
        return x[1:n]
    end
end
