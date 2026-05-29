# ------------------
# Augmented factorization (fallback path)
# ------------------
#
# Solve A x = b with A = S + U V via the equivalent bordered system
#
#     [ S   U ] [x]   [b]
#     [ V  -I ] [y] = [0]
#
# factored as ONE sparse LU of size (n+r). This requires only `A` (not `S`) to be
# nonsingular, so it is the correctness path when `S` is singular or the Woodbury
# correction `C` is dangerously ill-conditioned. The `r` dense rows/columns of `U`/`V` add
# fill comparable to FABM's QR cost — fine for small `r`, the regime where the low-rank
# structure is worth exploiting at all.

"""
    SparseWithDenseRowColAugmented{T} <: LinearAlgebra.Factorization{T}

Fallback factorization of an [`SparseWithDenseRowColMatrix`](@ref) `A = S + U V` via a single sparse
LU of the bordered system `[S U; V -I]` of size `n + r`. Used automatically by
[`factorize`](@ref) when `S` is singular or the Woodbury correction is ill-conditioned, and
selectable explicitly with `strategy = :augmented`. Requires only `A` (not `S`) to be
nonsingular.
"""
mutable struct SparseWithDenseRowColAugmented{T, KF} <: LinearAlgebra.Factorization{T}
    Kaug::KF             # PureKLU.KLUFactorization of [S U; V -I]
    n::Int
    r::Int
    rhs::Vector{T}       # length n+r work buffer ([b; 0] / solution)
end

Base.size(F::SparseWithDenseRowColAugmented) = (F.n, F.n)
Base.size(F::SparseWithDenseRowColAugmented, i::Integer) = i ≤ 2 ? F.n : 1
LinearAlgebra.issuccess(F::SparseWithDenseRowColAugmented) = LinearAlgebra.issuccess(F.Kaug)
denserank(F::SparseWithDenseRowColAugmented) = F.r

_sparse_block(U::AbstractMatrix{T}) where {T} = sparse(U)
function _sparse_block(U::SelectorMatrix{T}) where {T}
    r = U.r
    return SparseMatrixCSC(U.n, r, collect(1:(r + 1)), collect(1:r), ones(T, r))
end

function _augmented_matrix(A::SparseWithDenseRowColMatrix{T}) where {T}
    n = size(A, 1)
    r = denserank(A)
    S = _own_sparse(T, A.S)
    if r == 0
        return S
    end
    Usp = _sparse_block(A.U)               # n×r
    Vsp = sparse(Matrix{T}(A.V))           # r×n
    negI = SparseMatrixCSC{T, Int}(-one(T) * I, r, r)
    return [S Usp; Vsp negI]
end

function _augmented(A::SparseWithDenseRowColMatrix{T}) where {T}
    n = size(A, 1)
    r = denserank(A)
    Maug = _augmented_matrix(A)
    Kaug = PureKLU.klu(Maug)               # throws SingularException if A is singular
    return SparseWithDenseRowColAugmented{T, typeof(Kaug)}(Kaug, n, r, Vector{T}(undef, n + r))
end

function LinearAlgebra.ldiv!(F::SparseWithDenseRowColAugmented{T}, b::AbstractVector) where {T}
    length(b) == F.n || throw(DimensionMismatch())
    rhs = F.rhs
    @inbounds copyto!(view(rhs, 1:F.n), b)
    @inbounds for i in (F.n + 1):(F.n + F.r)
        rhs[i] = zero(T)
    end
    PureKLU.solve!(F.Kaug, rhs)
    @inbounds copyto!(b, view(rhs, 1:F.n))
    return b
end

function LinearAlgebra.ldiv!(F::SparseWithDenseRowColAugmented, B::AbstractMatrix)
    size(B, 1) == F.n || throw(DimensionMismatch())
    for c in axes(B, 2)
        ldiv!(F, view(B, :, c))
    end
    return B
end

LinearAlgebra.ldiv!(y::AbstractVector, F::SparseWithDenseRowColAugmented, b::AbstractVector) =
    ldiv!(F, copyto!(y, b))
LinearAlgebra.ldiv!(Y::AbstractMatrix, F::SparseWithDenseRowColAugmented, B::AbstractMatrix) =
    ldiv!(F, copyto!(Y, B))

# Adjoint / transpose solve. The transposed bordered system Mᵀ = [Sᵀ Vᵀ; Uᵀ -I] has Schur
# complement Sᵀ + Vᵀ Uᵀ = (S + U V)ᵀ = Aᵀ (and likewise Mᴴ ↔ Aᴴ), so solving Mᵀ[x;y]=[b;0]
# (resp. Mᴴ) with the SAME factorization's transpose/adjoint solve yields x = A⁻ᵀ b (resp. A⁻ᴴ b).
for (Wrap, kluop) in ((AdjointFact, :(F.Kaug')), (TransposeFact, :(transpose(F.Kaug))))
    @eval function LinearAlgebra.ldiv!(
            Fw::$Wrap{<:Any, <:SparseWithDenseRowColAugmented{T}}, b::AbstractVector
        ) where {T}
        F = parent(Fw)
        length(b) == F.n || throw(DimensionMismatch())
        rhs = F.rhs
        @inbounds copyto!(view(rhs, 1:F.n), b)
        @inbounds for i in (F.n + 1):(F.n + F.r)
            rhs[i] = zero(T)
        end
        PureKLU.solve!($kluop, rhs)
        @inbounds copyto!(b, view(rhs, 1:F.n))
        return b
    end
    @eval function LinearAlgebra.ldiv!(Fw::$Wrap{<:Any, <:SparseWithDenseRowColAugmented}, B::AbstractMatrix)
        for c in axes(B, 2)
            ldiv!(Fw, view(B, :, c))
        end
        return B
    end
end

# Complex RHS over a real augmented factorization (real/imag split; see woodbury.jl).
LinearAlgebra.ldiv!(F::SparseWithDenseRowColAugmented{<:Real}, b::AbstractVector{<:Complex}) =
    _ldiv_realfact_complex!(F, b)
LinearAlgebra.ldiv!(Fw::AdjointFact{<:Any, <:SparseWithDenseRowColAugmented{<:Real}}, b::AbstractVector{<:Complex}) =
    _ldiv_realfact_complex!(Fw, b)
LinearAlgebra.ldiv!(Fw::TransposeFact{<:Any, <:SparseWithDenseRowColAugmented{<:Real}}, b::AbstractVector{<:Complex}) =
    _ldiv_realfact_complex!(Fw, b)

"""
    refactor!(F::SparseWithDenseRowColAugmented, A::SparseWithDenseRowColMatrix; check=true) -> F

Rebuild the augmented system with new numeric values and refactor. When the augmented
sparsity pattern is unchanged this reuses PureKLU's symbolic analysis (`klu!`); otherwise it
re-analyzes. The augmented path is correctness-first, not the optimized refactor path.
"""
function refactor!(F::SparseWithDenseRowColAugmented{T}, A::SparseWithDenseRowColMatrix; check::Bool = true) where {T}
    denserank(A) == F.r && size(A, 1) == F.n ||
        throw(DimensionMismatch("shape changed; rebuild with `factorize`."))
    Maug = _augmented_matrix(A)
    if length(SparseArrays.nonzeros(Maug)) == length(F.Kaug.nzval) &&
            (
            !check || (
                SparseArrays.getcolptr(Maug) == increment(F.Kaug.colptr) &&
                    SparseArrays.rowvals(Maug) == increment(F.Kaug.rowval)
            )
        )
        PureKLU.klu!(F.Kaug, SparseArrays.nonzeros(Maug))
    else
        F.Kaug = PureKLU.klu(Maug)
    end
    return F
end

# 1-based copy of PureKLU's internal 0-based index arrays, for the pattern check above.
increment(v::AbstractVector{<:Integer}) = v .+ one(eltype(v))

# The augmented fallback embeds U/V inside one sparse LU and keeps no separate factorization
# of S to reuse, so the Woodbury low-rank fast path does not apply here.
function update_lowrank!(::SparseWithDenseRowColAugmented; kwargs...)
    throw(ArgumentError("update_lowrank! is the Woodbury fast path (it reuses S's factorization); \
        the augmented fallback (singular S) has no separate S factorization to reuse — \
        update with `refactor!` or rebuild with `factorize`."))
end
