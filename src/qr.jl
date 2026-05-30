# ------------------
# Augmented QR factorization (numerically stable path)
# ------------------
#
# Solve A x = b with A = S + U V via the bordered system
#
#     [ S   U ] [x]   [b]
#     [ V  -I ] [y] = [0]
#
# factored as ONE sparse QR (SuiteSparseQR, `qr(::SparseMatrixCSC)`) of size (n+r). This is
# the numerically stable analogue of the augmented LU path: it never forms `S⁻¹`, so it stays
# accurate even when the sparse part `S` is ill-conditioned or nearly singular but the dense
# rows/columns regularize `A` (the FastAlmostBandedMatrices regime). The bordered system is
# full-rank and well-conditioned whenever `A` itself is nonsingular, regardless of κ(S) — which
# is why a Woodbury-over-qr(S) approach is NOT shipped: it shares the κ(S)·κ(C) cancellation
# and is catastrophically wrong (~1e-1 forward error) once κ(S) ≳ 1e11, while this path holds
# near the noise floor across the whole range.

"""
    SparseWithDenseRowColQRAugmented{T} <: LinearAlgebra.Factorization{T}

QR factorization of a [`SparseWithDenseRowColMatrix`](@ref) `A = S + U V`, built as a single
sparse QR (SuiteSparseQR) of the bordered system `[S U; V -I]` of size `n + r`. This is the
numerically stable analogue of [`SparseWithDenseRowColAugmented`](@ref) (which uses LU): it
never forms `S⁻¹`, so it stays accurate even when `S` is ill-conditioned or nearly singular,
as long as `A` itself is nonsingular. Created by [`qr`](@ref).

Only `Float64` and `ComplexF64` are supported; for `Float32`/`ComplexF32` and all generic
eltypes use [`factorize`](@ref)/[`lu`](@ref) (PureKLU). (SuiteSparseQR's single-precision
sparse QR behaves inconsistently across Julia versions — it throws on 1.11 — so the QR path
restricts itself to the double-precision floats, which work identically everywhere.)

Unlike the LU path this path has no symbolic-reuse refactor (SuiteSparseQR exposes no `qr!`)
and no zero-allocation solve, so it is the *stability* path, not the *throughput* path. The
adjoint/transpose solve is supported via a separately built, lazily cached factorization of
the bordered system's adjoint.
"""
mutable struct SparseWithDenseRowColQRAugmented{T, QF} <: LinearAlgebra.Factorization{T}
    Qaug::QF                       # QRSparse of [S U; V -I]
    QaugH::Any                     # nothing until first adjoint/transpose solve, then a QRSparse
    Maug::SparseMatrixCSC{T, Int}  # owned bordered matrix (for refactor! and building QaugH)
    n::Int
    r::Int
    rankaug::Int                   # numerical rank of Qaug (== n+r ⇒ A nonsingular)
    rhs::Vector{T}                 # length n+r — [b; 0] / RHS scratch
    rsol::Vector{T}                # length n+r — receives `Qaug \ rhs` (QR has no in-place ldiv!)
end

Base.size(F::SparseWithDenseRowColQRAugmented) = (F.n, F.n)
Base.size(F::SparseWithDenseRowColQRAugmented, i::Integer) = i ≤ 2 ? F.n : 1
LinearAlgebra.issuccess(F::SparseWithDenseRowColQRAugmented) = F.rankaug == F.n + F.r
denserank(F::SparseWithDenseRowColQRAugmented) = F.r

# SuiteSparseQR's single-precision support is inconsistent across Julia versions (Float32/
# ComplexF32 sparse `qr` upcasts on 1.10, *throws* a CHOLMODException on 1.11, and is native on
# 1.12). To give identical behavior on every supported version we restrict the QR path to the
# double-precision BLAS floats, which work everywhere; Float32/ComplexF32 (and all generic
# eltypes) route to PureKLU's LU via `factorize`/`lu`.
_qr_supported(::Type{<:Union{Float64, ComplexF64}}) = true
_qr_supported(::Type) = false

function _check_qr_eltype(::Type{T}) where {T}
    _qr_supported(T) || throw(
        ArgumentError(
            "SparseWithDenseRowColQRAugmented (SuiteSparseQR) supports only Float64 and ComplexF64; " *
                "got eltype $T. Use `factorize`/`lu` (PureKLU) for Float32/ComplexF32 and generic eltypes " *
                "(SuiteSparseQR's single-precision sparse QR is unavailable on some Julia versions)."
        )
    )
    return nothing
end

# All `qr` calls below pass `tol = 0.0` to DISABLE SuiteSparseQR's default column deflation
# (`tol = -2` drops columns below ~20·(m+n)·ε·maxcolnorm and writes an exact 0 onto the
# R-diagonal). That default would report a merely ill-conditioned — but nonsingular and
# accurately solvable — bordered system as rank-deficient, making `qr(A)` throw on matrices
# that `factorize`/`klu` solve fine. With deflation off we make the rank decision ourselves
# from the R-diagonal, so the singularity cutoff matches the explicit threshold in `_qr_rank`
# rather than SuiteSparseQR's hidden one.
const _QR_NODEFLATE = 0.0

# SuiteSparseQR is rank-revealing; estimate rank from the R-diagonal against the standard
# threshold. `qr` does not throw on a rank-deficient bordered system (unlike `klu`), so this is
# how singularity of A is detected.
# The cutoff is `cond ≳ 1/(m·ε)` (~5e13 for Float64); genuine rank deficiency yields an
# exact-zero pivot and is caught regardless. A truly singular bulk is reported here as a
# `SingularException` (see `_augmented_qr`), matching the LU path's contract.
function _qr_rank(Fq, m::Int)
    d = abs.(diag(Fq.R))
    isempty(d) && return 0
    tol = maximum(d) * m * eps(real(eltype(Fq)))
    return count(>(tol), d)
end

function _augmented_qr(A::SparseWithDenseRowColMatrix{T}) where {T}
    _check_qr_eltype(T)
    n = size(A, 1)
    r = denserank(A)
    Maug = SparseMatrixCSC{T, Int}(_augmented_matrix(A))
    Qaug = qr(Maug; tol = _QR_NODEFLATE)
    rankaug = _qr_rank(Qaug, n + r)
    rankaug == n + r || throw(SingularException(0))   # A singular ⇒ bordered system singular
    return SparseWithDenseRowColQRAugmented{T, typeof(Qaug)}(
        Qaug, nothing, Maug, n, r, rankaug,
        Vector{T}(undef, n + r), Vector{T}(undef, n + r)
    )
end

function LinearAlgebra.ldiv!(F::SparseWithDenseRowColQRAugmented{T}, b::AbstractVector) where {T}
    length(b) == F.n || throw(DimensionMismatch())
    rhs = F.rhs
    @inbounds copyto!(view(rhs, 1:F.n), b)
    @inbounds for i in (F.n + 1):(F.n + F.r)
        rhs[i] = zero(T)
    end
    copyto!(F.rsol, F.Qaug \ rhs)          # out-of-place QR solve (allocates the solution)
    @inbounds copyto!(b, view(F.rsol, 1:F.n))
    return b
end

function LinearAlgebra.ldiv!(F::SparseWithDenseRowColQRAugmented, B::AbstractMatrix)
    size(B, 1) == F.n || throw(DimensionMismatch())
    for c in axes(B, 2)
        ldiv!(F, view(B, :, c))
    end
    return B
end

LinearAlgebra.ldiv!(y::AbstractVector, F::SparseWithDenseRowColQRAugmented, b::AbstractVector) =
    ldiv!(F, copyto!(y, b))
LinearAlgebra.ldiv!(Y::AbstractMatrix, F::SparseWithDenseRowColQRAugmented, B::AbstractMatrix) =
    ldiv!(F, copyto!(Y, B))

# Adjoint / transpose solve. The bordered matrix Mᴴ = [Sᴴ Vᴴ; Uᴴ -I] has Schur complement
# Sᴴ + Vᴴ Uᴴ = (S + U V)ᴴ = Aᴴ (and Mᵀ ↔ Aᵀ), so solving Mᴴ[x;y]=[b;0] gives x = A⁻ᴴ b. Unlike
# the LU path, SuiteSparseQR has no `Qaug' \ b`, so we build a SEPARATE qr of the adjoint and
# cache it lazily (it is invalidated on every `refactor!`).
function _augfact_adj!(F::SparseWithDenseRowColQRAugmented)
    F.QaugH === nothing && (F.QaugH = qr(sparse(F.Maug'); tol = _QR_NODEFLATE))
    return F.QaugH
end
# For real T, transpose == adjoint, so reuse the cached factorization; for complex T the
# (rare) transpose path rebuilds qr(transpose(Maug)) each call rather than holding a 3rd cache.
_augfact_tr!(F::SparseWithDenseRowColQRAugmented{<:Real}) = _augfact_adj!(F)
_augfact_tr!(F::SparseWithDenseRowColQRAugmented) = qr(sparse(transpose(F.Maug)); tol = _QR_NODEFLATE)

for (Wrap, getfact) in ((AdjointFact, :_augfact_adj!), (TransposeFact, :_augfact_tr!))
    @eval function LinearAlgebra.ldiv!(
            Fw::$Wrap{<:Any, <:SparseWithDenseRowColQRAugmented{T}}, b::AbstractVector
        ) where {T}
        F = parent(Fw)
        length(b) == F.n || throw(DimensionMismatch())
        rhs = F.rhs
        @inbounds copyto!(view(rhs, 1:F.n), b)
        @inbounds for i in (F.n + 1):(F.n + F.r)
            rhs[i] = zero(T)
        end
        copyto!(F.rsol, $getfact(F) \ rhs)
        @inbounds copyto!(b, view(F.rsol, 1:F.n))
        return b
    end
    @eval function LinearAlgebra.ldiv!(Fw::$Wrap{<:Any, <:SparseWithDenseRowColQRAugmented}, B::AbstractMatrix)
        for c in axes(B, 2)
            ldiv!(Fw, view(B, :, c))
        end
        return B
    end
end

# Complex RHS over a real augmented-QR factorization (real/imag split; see woodbury.jl).
LinearAlgebra.ldiv!(F::SparseWithDenseRowColQRAugmented{<:Real}, b::AbstractVector{<:Complex}) =
    _ldiv_realfact_complex!(F, b)
LinearAlgebra.ldiv!(Fw::AdjointFact{<:Any, <:SparseWithDenseRowColQRAugmented{<:Real}}, b::AbstractVector{<:Complex}) =
    _ldiv_realfact_complex!(Fw, b)
LinearAlgebra.ldiv!(Fw::TransposeFact{<:Any, <:SparseWithDenseRowColQRAugmented{<:Real}}, b::AbstractVector{<:Complex}) =
    _ldiv_realfact_complex!(Fw, b)

"""
    refactor!(F::SparseWithDenseRowColQRAugmented, A::SparseWithDenseRowColMatrix; check=true) -> F

Rebuild the augmented system with the new values of `A` and re-factor. Unlike the LU paths
there is **no symbolic-reuse fast path** — SuiteSparseQR exposes no `qr!`, so this always
re-`qr`s the full bordered system. For hot refactor loops (Newton / time stepping) prefer
[`factorize`](@ref)/[`lu`](@ref), which reuse PureKLU's symbolic analysis; the QR path is the
stability path, not the throughput path. The shape (`n`, `r`) must be unchanged.
"""
function refactor!(F::SparseWithDenseRowColQRAugmented{T}, A::SparseWithDenseRowColMatrix; check::Bool = true) where {T}
    denserank(A) == F.r && size(A, 1) == F.n ||
        throw(DimensionMismatch("shape changed; rebuild with `qr`."))
    F.Maug = SparseMatrixCSC{T, Int}(_augmented_matrix(A))
    F.Qaug = qr(F.Maug; tol = _QR_NODEFLATE)
    F.QaugH = nothing                                     # invalidate cached adjoint
    F.rankaug = _qr_rank(F.Qaug, F.n + F.r)
    check && (F.rankaug == F.n + F.r || throw(SingularException(0)))
    return F
end

# The augmented-QR path embeds U/V inside one factorization and keeps no separate factorization
# of S to reuse, so the Woodbury low-rank fast path does not apply here (same as augmented LU).
function update_lowrank!(::SparseWithDenseRowColQRAugmented; kwargs...)
    throw(ArgumentError("update_lowrank! is the Woodbury fast path (it reuses S's factorization); \
        the augmented QR path embeds U/V in one factorization and has no separate S factorization \
        to reuse — update with `refactor!` or rebuild with `qr`."))
end
