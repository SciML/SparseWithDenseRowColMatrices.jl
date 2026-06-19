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

# Sparse "peel" for a SINGULAR S whose null structure is coordinate-aligned (the BVP /
# boundary-condition case: `replace=true` zeros the selector rows of S, U = [I_r; 0]). The
# identity A = S + U V = (S + U Uᴴ) + U(V − Uᴴ) re-splits A with S̃ = S + U Uᴴ; for a selector U,
# U Uᴴ is just `r` diagonal entries, so S̃ stays SPARSE and is typically nonsingular (the zeroed
# rows get a unit diagonal back), and the rank is unchanged. The structured method then applies
# to (S̃, U, V − Uᴴ). For a dense U, U Uᴴ is dense (S̃ dense) so no sparse peel exists — return
# `nothing` and let the caller fall back to the dense COD.
_peel_selector(S, U, V) = nothing
function _peel_selector(S::SparseMatrixCSC{T}, U::SelectorMatrix, V::AbstractMatrix{T}) where {T}
    r = U.r
    r == 0 && return nothing
    n = size(S, 1)
    Stilde = S + sparse(1:r, 1:r, ones(T, r), n, n)        # S̃ = S + U Uᴴ  (sparse)
    Vtilde = copy(V)
    @inbounds for k in 1:r
        Vtilde[k, k] -= one(T)                              # Ṽ = V − Uᴴ
    end
    return (Stilde, Vtilde)
end

# Factor S; on (exact) singularity try the sparse selector-peel and factor S̃ instead. Returns
# `(Sfact, S_effective, V_effective)` or `nothing` (caller falls back to the dense COD).
# PureKLU does not throw on a singular factor — it stops at the zero pivot and reports
# `issuccess == false` — so a successful factorization is detected via `issuccess`, not the
# absence of a SingularException.
function _klu_or_peel(Sown::SparseMatrixCSC, U, Vmat)
    Sfact = PureKLU.klu(Sown)
    LinearAlgebra.issuccess(Sfact) && return (Sfact, Sown, Vmat)
    peeled = _peel_selector(Sown, U, Vmat)
    peeled === nothing && return nothing
    Stilde, Vtilde = peeled
    Stildefact = PureKLU.klu(Stilde)
    LinearAlgebra.issuccess(Stildefact) || return nothing   # S̃ still singular → dense fallback
    return (Stildefact, Stilde, Vtilde)
end

# Structure-exploiting DIRECT min-norm least-squares (returns `(x, ok)`; `ok=false` signals the
# caller to fall back to dense because S is singular/near-singular). With S nonsingular,
# A = S(I + ZV), Z = S⁻¹U; the rank deficiency lives in C = I + V Z (nullity(A)=nullity(C)).
# A⁺b is assembled from one PureKLU factorization of S + small dense SVD/QR — A is never formed.
# A SINGULAR S with coordinate-aligned null structure (selector U) is first peeled to a sparse
# nonsingular S̃ (see `_peel_selector`); a general singular S falls back to the dense COD.
"""
    SparseWithDenseRowColLeastSquares{T} <: LinearAlgebra.Factorization{T}

Cached structure-exploiting least-squares factorization of an [`SparseWithDenseRowColMatrix`](@ref)
`A = S + U V`, for **repeated** minimum-norm least-squares solves (e.g. a Newton / time-stepping
loop): the expensive work — the sparse factorization of `S` (PureKLU) and the small dense
SVD/QR — is done once at construction, so each `F \\ b` / `ldiv!(F, b)` is a cheap back-solve
that **reuses** it. `F \\ b` returns `A⁺b` exactly as [`lstsq`](@ref)`(A, b)` does.

Constructed by `SparseWithDenseRowColLeastSquares(A)`. Only valid where the structured direct
method applies (`S` nonsingular, or singular with a coordinate-aligned null space); it errors
otherwise — use `lstsq(A, b; alg = :dense)` for a general singular `S`.
"""
struct SparseWithDenseRowColLeastSquares{T, FT} <: LinearAlgebra.Factorization{T}
    Sfact::FT             # PureKLU factorization of the (possibly peeled) S̃
    W::Matrix{T}          # n × s active subspace basis
    Msp::Matrix{T}        # s × s, the assembled Mᴴ⁺ on the active block
    QL::Matrix{T}         # n × kdef left-null basis (range-of-A projector); n × 0 if full rank
    n::Int
    r::Int
    kdef::Int
    cbuf::Vector{T}       # length-n scratch reused across solves
end

Base.size(F::SparseWithDenseRowColLeastSquares) = (F.n, F.n)
Base.size(F::SparseWithDenseRowColLeastSquares, i::Integer) = i ≤ 2 ? F.n : 1
denserank(F::SparseWithDenseRowColLeastSquares) = F.r

# Expensive SETUP: factor S (or peel a selector-singular S̃) and build the small dense
# decompositions. Returns the cached factorization, or `nothing` to signal "fall back to dense"
# (S singular/near-singular with a non-sparse null space). Used by both the one-shot `lstsq`
# and the reusable factorization object.
function _structured_setup(A::SparseWithDenseRowColMatrix, ::Type{RT}; tolC::Real) where {RT}
    TT = float(RT)
    rT = real(TT)
    n = size(A, 1)
    r = denserank(A)
    tol = tolC < 0 ? sqrt(eps(rT)) : rT(tolC)

    Sown = _own_sparse(TT, A.S)
    Vmat = r > 0 ? Matrix{TT}(A.V) : Matrix{TT}(undef, 0, n)
    fac = _klu_or_peel(Sown, A.U, Vmat)             # factor S, or peel a selector-singular S to S̃
    fac === nothing && return nothing               # S (and any peel) singular → fall back to dense
    Sfact, Sown, V = fac                            # the *effective* (possibly peeled) factors

    empty_n0 = Matrix{TT}(undef, n, 0)
    cbuf = Vector{TT}(undef, n)
    r == 0 && return SparseWithDenseRowColLeastSquares{TT, typeof(Sfact)}(   # A = S nonsingular
        Sfact, empty_n0, Matrix{TT}(undef, 0, 0), empty_n0, n, 0, 0, cbuf
    )

    Z = Matrix{TT}(undef, n, r)
    materialize_U!(Z, A.U)                          # Z = U
    nrmU = norm(Z)
    PureKLU.solve!(Sfact, Z)                        # Z = S̃⁻¹U

    # Guard: near-singular S makes Z = S⁻¹U huge/garbage and `klu` does NOT throw, which both
    # crashes the SVDs and gives a silently-wrong answer. κ̂(S) = ‖S‖₁·‖Z‖/‖U‖ ≳ κ(S); bail early.
    kappaS = nrmU > 0 ? opnorm(Sown, 1) * norm(Z) / nrmU : zero(rT)
    (all(isfinite, Z) && isfinite(kappaS) && kappaS ≤ inv(sqrt(eps(rT)))) || return nothing

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
    Msp = Fs.V * Diagonal(sp) * Fs.U'               # s × s assembled Mᴴ⁺ on the active block

    QL = if kdef > 0
        Nl = Fc.U[:, (end - kdef + 1):end]          # null(Cᴴ)  (r × kdef)
        LrA = Vadj * Nl                             # n × kdef
        PureKLU.solve!(Sfact', LrA)                 # left-null(A) = S⁻ᴴ Vᴴ null(Cᴴ)
        Matrix(qr(LrA).Q)[:, 1:kdef]
    else
        empty_n0
    end
    return SparseWithDenseRowColLeastSquares{TT, typeof(Sfact)}(Sfact, W, Msp, QL, n, r, kdef, cbuf)
end

# Cheap APPLY: x = A⁺b from the cached factorization (no re-factorization of S). Reads `b` fully
# before writing `x`, so `x === b` (in-place) is allowed.
function _structured_apply!(x::AbstractVector, F::SparseWithDenseRowColLeastSquares{T}, b::AbstractVector) where {T}
    c = F.cbuf
    copyto!(c, b)
    if F.kdef > 0                                   # project b onto range(A): c = b − QL(QLᴴ b)
        mul!(c, F.QL, F.QL' * b, -one(T), one(T))
    end
    PureKLU.solve!(F.Sfact, c)                       # c = S̃⁻¹ b̃
    if size(F.W, 2) > 0
        Wtc = F.W' * c                              # s-vector
        copyto!(x, c)
        mul!(x, F.W, Wtc, -one(T), one(T))          # x = c − W Wᴴc
        mul!(x, F.W, F.Msp * Wtc, one(T), one(T))   # x += W Mᴴ⁺ Wᴴc
    else
        copyto!(x, c)
    end
    return x
end

# One-shot structured solve (returns `(x, ok)`; `ok=false` ⇒ caller falls back to dense). Setup +
# apply + a post-hoc least-squares-optimality check on the actual `b` (catches a misjudged rank).
function _lstsq_structured(
        A::SparseWithDenseRowColMatrix, b::AbstractVector;
        tolC::Real = -one(real(float(eltype(A)))), λ::Real = 0,
        # accept and ignore the other engines' keywords so `alg` can be flipped without edits
        rcond = nothing, solver = nothing, atol = nothing, btol = nothing,
        maxiters = nothing, verbose = nothing,
    )
    n = size(A, 1)
    length(b) == n || throw(DimensionMismatch("A has $n columns, b has length $(length(b))"))
    λ ≥ 0 || throw(ArgumentError("λ (Tikhonov damping) must be ≥ 0; got $λ"))
    iszero(λ) || throw(ArgumentError("lstsq(...; alg=:structured) does not support Tikhonov λ>0; use alg=:dense."))
    RT = promote_type(eltype(A), eltype(b))
    _check_lstsq_eltype(RT)
    TT = float(RT)
    F = _structured_setup(A, RT; tolC = tolC)
    F === nothing && return (zeros(TT, n), false)
    x = _structured_apply!(Vector{TT}(undef, n), F, collect(TT, b))

    # post-hoc least-squares optimality: ‖Aᴴ(Ax−b)‖ → 0 at a minimizer; a loose threshold
    # separates a good solution (~1e-14 relative) from a grossly wrong one. A false trip only
    # falls `:auto` back to the exact dense path, so erring conservative is safe.
    res = A * x
    res .-= b
    atr = A' * res
    nrm_atr = norm(atr)
    ok = nrm_atr ≤ 1.0e-6 * (norm(A' * collect(TT, b)) + nrm_atr)
    return (x, ok)
end

"""
    SparseWithDenseRowColLeastSquares(A::SparseWithDenseRowColMatrix; tolC=…) -> F

Build a reusable structured least-squares factorization for repeated `A⁺b` solves (see
[`SparseWithDenseRowColLeastSquares`](@ref)). Errors if the structured direct method does not
apply to `A` (general singular/near-singular `S`); use `lstsq(A, b; alg = :dense)` there.
"""
function SparseWithDenseRowColLeastSquares(A::SparseWithDenseRowColMatrix; tolC::Real = -one(real(float(eltype(A)))))
    _check_lstsq_eltype(eltype(A))
    F = _structured_setup(A, eltype(A); tolC = tolC)
    F === nothing && throw(
        ArgumentError(
            "a structured least-squares factorization does not apply to this A (S is singular or " *
                "near-singular with a non-sparse null space); use `lstsq(A, b; alg = :dense)`."
        )
    )
    return F
end

LinearAlgebra.ldiv!(F::SparseWithDenseRowColLeastSquares, b::AbstractVector) =
    (length(b) == F.n || throw(DimensionMismatch()); _structured_apply!(b, F, b))
LinearAlgebra.ldiv!(x::AbstractVector, F::SparseWithDenseRowColLeastSquares, b::AbstractVector) =
    (length(b) == F.n || throw(DimensionMismatch()); _structured_apply!(x, F, b))
function Base.:\(F::SparseWithDenseRowColLeastSquares{T}, b::AbstractVector) where {T}
    length(b) == F.n || throw(DimensionMismatch())
    return _structured_apply!(Vector{T}(undef, F.n), F, eltype(b) === T ? b : convert(Vector{T}, b))
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
