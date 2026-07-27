# ------------------
# Augmented QR factorization (numerically stable path)
# ------------------
#
# Solve A x = b with A = S + U V via the bordered system
#
#     [ S   U ] [x]   [b]
#     [ V  -I ] [y] = [0]
#
# factored as ONE sparse, rank-revealing, column-pivoted QR (SparseColumnPivotedQR's `scpqr`)
# of size (n+r). This is the numerically stable analogue of the augmented LU path: it never
# forms `S⁻¹`, so it stays accurate even when the sparse part `S` is ill-conditioned or nearly
# singular but the dense rows/columns regularize `A` (the FastAlmostBandedMatrices regime). The
# bordered system is full-rank and well-conditioned whenever `A` itself is nonsingular,
# regardless of κ(S) — which is why a Woodbury-over-qr(S) approach is NOT shipped: it shares the
# κ(S)·κ(C) cancellation and is catastrophically wrong once κ(S) ≳ 1e11, while this path holds
# near the noise floor across the whole range.
#
# Backend: SparseColumnPivotedQR.jl (pure Julia, like PureKLU is for LU). Versus a SuiteSparseQR
# backend this gives (a) an allocation-free in-place solve, (b) a symbolic-reuse `scpqr_refactor!`
# for the Newton/time-stepping hot path, (c) genuine column-pivoted rank revelation (so a
# singular `A` is detected directly from `rank`), and (d) generic element types.

"""
    SparseWithDenseRowColQRAugmented{T} <: LinearAlgebra.Factorization{T}

QR factorization of a [`SparseWithDenseRowColMatrix`](@ref) `A = S + U V`, built as a single
rank-revealing, column-pivoted sparse QR (SparseColumnPivotedQR) of the bordered system
`[S U; V -I]` of size `n + r`. This is the numerically stable analogue of
[`SparseWithDenseRowColAugmented`](@ref) (which uses LU): it never forms `S⁻¹`, so it stays
accurate even when `S` is ill-conditioned or nearly singular, as long as `A` itself is
nonsingular. Created by [`qr`](@ref LinearAlgebra.qr).

The solve is allocation-free, [`refactor!`](@ref) reuses the symbolic analysis (so it is the
throughput path for fixed-pattern Newton loops as well as the stability path), and any element
type the backend supports is allowed (the BLAS floats plus generic numbers such as `BigFloat`
and `ForwardDiff.Dual`). The adjoint/transpose solve uses a separately built, lazily cached
factorization of the bordered system's adjoint.
"""
mutable struct SparseWithDenseRowColQRAugmented{T, QF} <: LinearAlgebra.Factorization{T}
    Qaug::QF                       # SparseColumnPivotedQRFactorization of [S U; V -I]
    QaugH::Any                     # nothing until first adjoint/transpose solve, then a SparseColumnPivotedQRFactorization
    MaugC::SparseMatrixCSC{T, Int} # owned bordered matrix (CSC; source for refactor! and the adjoint)
    n::Int
    r::Int
    rankaug::Int                   # numerical rank of Qaug (== n+r ⇒ A nonsingular)
    rhs::Vector{T}                 # length n+r — [b; 0] / RHS scratch
    rsol::Vector{T}                # length n+r — receives the in-place solve
end

Base.size(F::SparseWithDenseRowColQRAugmented) = (F.n, F.n)
Base.size(F::SparseWithDenseRowColQRAugmented, i::Integer) = i ≤ 2 ? F.n : 1
LinearAlgebra.issuccess(F::SparseWithDenseRowColQRAugmented) = F.rankaug == F.n + F.r
denserank(F::SparseWithDenseRowColQRAugmented) = F.r
Base.adjoint(F::SparseWithDenseRowColQRAugmented) = _adjoint_factorization(F)
Base.transpose(F::SparseWithDenseRowColQRAugmented) = _transpose_factorization(F)

# CSC of Mᴴ / Mᵀ (for the adjoint/transpose factorization), built from the owned CSC.
_adjoint_csc(MaugC::SparseMatrixCSC{T}) where {T} = SparseMatrixCSC{T, Int}(MaugC')
_transpose_csc(MaugC::SparseMatrixCSC{T}) where {T} = SparseMatrixCSC{T, Int}(transpose(MaugC))

function _augmented_qr(A::SparseWithDenseRowColMatrix{T}) where {T}
    n = size(A, 1)
    r = denserank(A)
    MaugC = SparseMatrixCSC{T, Int}(_augmented_matrix(A))
    Qaug = SparseColumnPivotedQR.scpqr(MaugC)
    rankaug = LinearAlgebra.rank(Qaug)
    rankaug == n + r || throw(SingularException(0))   # A singular ⇒ bordered system singular
    return SparseWithDenseRowColQRAugmented{T, typeof(Qaug)}(
        Qaug, nothing, MaugC, n, r, rankaug,
        Vector{T}(undef, n + r), Vector{T}(undef, n + r)
    )
end

# Fill `F.rhs = [b; 0]`, solve in place into `F.rsol` with `fact`, copy the leading `n` back.
@inline function _qr_aug_solve!(F::SparseWithDenseRowColQRAugmented{T}, fact, b::AbstractVector) where {T}
    @inbounds copyto!(view(F.rhs, 1:F.n), b)
    @inbounds for i in (F.n + 1):(F.n + F.r)
        F.rhs[i] = zero(T)
    end
    LinearAlgebra.ldiv!(F.rsol, fact, F.rhs)          # allocation-free in-place solve
    @inbounds copyto!(b, view(F.rsol, 1:F.n))
    return b
end

function LinearAlgebra.ldiv!(F::SparseWithDenseRowColQRAugmented, b::AbstractVector)
    length(b) == F.n || throw(DimensionMismatch())
    return _qr_aug_solve!(F, F.Qaug, b)
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
# Sᴴ + Vᴴ Uᴴ = (S + U V)ᴴ = Aᴴ (and Mᵀ ↔ Aᵀ), so solving Mᴴ[x;y]=[b;0] gives x = A⁻ᴴ b. The
# column-pivoted QR has no `Qaug' \ b`, so we build a SEPARATE factorization of the adjoint and
# cache it lazily (it is invalidated on every `refactor!`).
function _augfact_adj!(F::SparseWithDenseRowColQRAugmented)
    F.QaugH === nothing && (F.QaugH = SparseColumnPivotedQR.scpqr(_adjoint_csc(F.MaugC)))
    return F.QaugH
end
# For real T, transpose == adjoint, so reuse the cached factorization; for complex T the
# (rare) transpose path rebuilds qr(transpose(Maug)) each call rather than holding a 3rd cache.
_augfact_tr!(F::SparseWithDenseRowColQRAugmented{<:Real}) = _augfact_adj!(F)
_augfact_tr!(F::SparseWithDenseRowColQRAugmented) = SparseColumnPivotedQR.scpqr(_transpose_csc(F.MaugC))

for (Wrap, getfact) in ((_AdjointFactorization, :_augfact_adj!), (_TransposeFactorization, :_augfact_tr!))
    @eval function LinearAlgebra.ldiv!(
            Fw::$Wrap{<:Any, <:SparseWithDenseRowColQRAugmented}, b::AbstractVector
        )
        F = parent(Fw)
        length(b) == F.n || throw(DimensionMismatch())
        return _qr_aug_solve!(F, $getfact(F), b)
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
LinearAlgebra.ldiv!(Fw::_AdjointFactorization{<:Any, <:SparseWithDenseRowColQRAugmented{<:Real}}, b::AbstractVector{<:Complex}) =
    _ldiv_realfact_complex!(Fw, b)
LinearAlgebra.ldiv!(Fw::_TransposeFactorization{<:Any, <:SparseWithDenseRowColQRAugmented{<:Real}}, b::AbstractVector{<:Complex}) =
    _ldiv_realfact_complex!(Fw, b)

"""
    refactor!(F::SparseWithDenseRowColQRAugmented, A::SparseWithDenseRowColMatrix; check=true) -> F

Rebuild the augmented system with the new values of `A` and re-factor in place. When the
bordered sparsity pattern is unchanged this **reuses the symbolic analysis** (column ordering,
elimination tree) and the preallocated numeric workspace — far cheaper than a fresh
factorization — making the QR path usable in a Newton / time-stepping hot loop, not just for
one-off stable solves. (The re-factorization kernel itself is allocation-free; this wrapper
still rebuilds the bordered matrix each call, so it is not yet fully zero-allocation.) The shape
(`n`, `r`) must be unchanged.
"""
function refactor!(F::SparseWithDenseRowColQRAugmented{T}, A::SparseWithDenseRowColMatrix; check::Bool = true) where {T}
    denserank(A) == F.r && size(A, 1) == F.n ||
        throw(DimensionMismatch("shape changed; rebuild with `qr`."))
    F.MaugC = SparseMatrixCSC{T, Int}(_augmented_matrix(A))
    SparseColumnPivotedQR.scpqr_refactor!(F.Qaug, F.MaugC)   # reuses the symbolic analysis
    F.QaugH = nothing                                                      # invalidate cached adjoint
    F.rankaug = LinearAlgebra.rank(F.Qaug)
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
