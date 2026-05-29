# ------------------
# SparseWithDenseRowColMatrix
# ------------------

"""
    SparseWithDenseRowColMatrix(S::AbstractSparseMatrixCSC, U::AbstractMatrix, V::AbstractMatrix)
    SparseWithDenseRowColMatrix(S::AbstractSparseMatrixCSC, fill::AbstractMatrix; replace=false)
    SparseWithDenseRowColMatrix{T}(S, U, V)

A square matrix represented as a **sparse part plus a low-rank dense correction**

    A = S + U * V

where `S` is an `n × n` `SparseMatrixCSC`, `U` is `n × r` and `V` is `r × n` dense, with
`r ≪ n` the *almost-sparse rank*. This is the sparse-matrix analogue of an
`AlmostBandedMatrix`: the banded "bulk" is replaced by a general sparse bulk, and the
low-rank `fill` (which in the banded case sits in the top rows as boundary conditions) is
kept as the explicit outer product `U * V` rather than folded into a band.

The structure is exactly the one the Sherman–Morrison–Woodbury identity exploits, so
[`factorize`](@ref)/`\\` solve `A x = b` by factoring only the sparse `S` (with
[`PureKLU`](https://github.com/SciML/PureKLU.jl)) and applying a small `r × r` dense
correction — see [`SparseWithDenseRowColWoodbury`](@ref). When `S` is singular but `A` is not, an
[`SparseWithDenseRowColAugmented`](@ref) fallback is used automatically.

# Constructors

* `SparseWithDenseRowColMatrix(S, U, V)` — the general additive low-rank form `A = S + U*V`.
* `SparseWithDenseRowColMatrix(S, fill)` — the boundary-condition convenience: `fill` is an `r × n`
  dense block placed in the **top `r` rows** via an implicit [`SelectorMatrix`](@ref)
  `U = [I_r; 0]`.
    * `replace = false` (default): `fill` is *added* to the top `r` rows, `A = S + [I_r;0]*fill`.
    * `replace = true`: the top `r` rows of `A` *become* `fill` (FABM-style row replacement),
      implemented as `V = fill - S[1:r, :]` so that `S` itself is left untouched and stays
      nonsingular for the Woodbury path.

```jldoctest
julia> using SparseWithDenseRowColMatrices, SparseArrays

julia> S = sparse(1.0I, 4, 4); fill = reshape(Float64[1,2,3,4], 1, 4);

julia> A = SparseWithDenseRowColMatrix(S, fill);   # rank-1 correction in row 1

julia> Matrix(A)[1, :]
4-element Vector{Float64}:
 2.0
 2.0
 3.0
 4.0
```
"""
struct SparseWithDenseRowColMatrix{
        T,
        TS <: SparseArrays.AbstractSparseMatrixCSC{T},
        TU <: AbstractMatrix{T},
        TV <: AbstractMatrix{T},
    } <: AbstractMatrix{T}
    S::TS
    U::TU
    V::TV

    function SparseWithDenseRowColMatrix{T, TS, TU, TV}(S, U, V) where {T, TS, TU, TV}
        n = size(S, 1)
        size(S, 2) == n ||
            throw(ArgumentError("the sparse part of an SparseWithDenseRowColMatrix must be square; got size $(size(S))"))
        size(U, 1) == n ||
            throw(DimensionMismatch("size(U, 1) = $(size(U, 1)) must equal n = $n"))
        size(V, 2) == n ||
            throw(DimensionMismatch("size(V, 2) = $(size(V, 2)) must equal n = $n"))
        size(U, 2) == size(V, 1) ||
            throw(DimensionMismatch("low-rank factor mismatch: size(U, 2) = $(size(U, 2)) ≠ size(V, 1) = $(size(V, 1))"))
        return new{T, TS, TU, TV}(S, U, V)
    end
end

_to_eltype(::Type{T}, A::AbstractMatrix{T}) where {T} = A
_to_eltype(::Type{T}, A::AbstractMatrix) where {T} = convert(AbstractMatrix{T}, A)
_to_eltype(::Type{T}, A::SelectorMatrix) where {T} = SelectorMatrix{T}(A.n, A.r)

function SparseWithDenseRowColMatrix(
        S::SparseArrays.AbstractSparseMatrixCSC, U::AbstractMatrix, V::AbstractMatrix
    )
    T = promote_type(eltype(S), eltype(U), eltype(V))
    return SparseWithDenseRowColMatrix{T}(S, U, V)
end

function SparseWithDenseRowColMatrix{T}(
        S::SparseArrays.AbstractSparseMatrixCSC, U::AbstractMatrix, V::AbstractMatrix
    ) where {T}
    Sc, Uc, Vc = _to_eltype(T, S), _to_eltype(T, U), _to_eltype(T, V)
    return SparseWithDenseRowColMatrix{T, typeof(Sc), typeof(Uc), typeof(Vc)}(Sc, Uc, Vc)
end

# Boundary-condition convenience: `fill` is an r×n block in the top r rows via a selector.
function SparseWithDenseRowColMatrix(
        S::SparseArrays.AbstractSparseMatrixCSC, fill::AbstractMatrix; replace::Bool = false
    )
    n = size(S, 2)
    r = size(fill, 1)
    size(fill, 2) == n ||
        throw(DimensionMismatch("fill has $(size(fill, 2)) columns but the sparse part has $n"))
    T = promote_type(eltype(S), eltype(fill))
    U = SelectorMatrix{T}(n, r)
    V = if replace
        # A's top r rows become `fill`; leave S (hence its nonsingularity) untouched by
        # absorbing the original top rows into the correction.
        Vd = Matrix{T}(fill)
        @inbounds for j in 1:n, i in 1:r
            Vd[i, j] -= S[i, j]
        end
        Vd
    else
        _to_eltype(T, fill)
    end
    return SparseWithDenseRowColMatrix{T}(S, U, V)
end

# ------------------
# Accessors (FABM parity)
# ------------------

"""
    sparsepart(A::SparseWithDenseRowColMatrix)

The sparse bulk `S` (the analogue of `bandpart` for an `AlmostBandedMatrix`).
"""
@inline sparsepart(A::SparseWithDenseRowColMatrix) = A.S

"""
    fillpart(A::SparseWithDenseRowColMatrix)

The `r × n` right low-rank factor `V` — the dense "fill" rows (the analogue of `fillpart`
for an `AlmostBandedMatrix`).
"""
@inline fillpart(A::SparseWithDenseRowColMatrix) = A.V

"""
    lowrankfactors(A::SparseWithDenseRowColMatrix) -> (U, V)

The pair `(U, V)` whose product `U * V` is the low-rank correction added to the sparse part.
"""
@inline lowrankfactors(A::SparseWithDenseRowColMatrix) = (A.U, A.V)

"""
    denserank(A::SparseWithDenseRowColMatrix)

The rank `r` of the low-rank correction (analogue of `almostbandedrank`).
"""
@inline denserank(A::SparseWithDenseRowColMatrix) = size(A.V, 1)

"""
    exclusive_sparsepart(A::SparseWithDenseRowColMatrix)

A view of the rows of the sparse part that the low-rank correction does **not** touch.
Only defined when the left factor is a [`SelectorMatrix`](@ref) (the boundary-condition
case), where the correction is confined to the top `r` rows; otherwise the correction is
not row-localized and this throws.
"""
function exclusive_sparsepart(A::SparseWithDenseRowColMatrix)
    A.U isa SelectorMatrix ||
        throw(ArgumentError("exclusive_sparsepart is only defined when the low-rank left factor is a SelectorMatrix; use `lowrankfactors` instead."))
    r = denserank(A)
    return @view A.S[(r + 1):end, :]
end

# ------------------
# AbstractArray interface
# ------------------

@inline Base.size(A::SparseWithDenseRowColMatrix) = size(A.S)
@inline Base.eltype(::SparseWithDenseRowColMatrix{T}) where {T} = T
Base.IndexStyle(::Type{<:SparseWithDenseRowColMatrix}) = IndexCartesian()

@inline _lowrank_entry(U::SelectorMatrix, V, i::Int, j::Int) =
    i ≤ U.r ? @inbounds(V[i, j]) : zero(eltype(V))
@inline function _lowrank_entry(U, V, i::Int, j::Int)
    acc = zero(promote_type(eltype(U), eltype(V)))
    @inbounds for k in 1:size(U, 2)
        acc += U[i, k] * V[k, j]
    end
    return acc
end

Base.@propagate_inbounds function Base.getindex(A::SparseWithDenseRowColMatrix, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    return @inbounds A.S[i, j] + _lowrank_entry(A.U, A.V, i, j)
end

"""
    setindex!(A::SparseWithDenseRowColMatrix, v, i, j)

Set the assembled entry `A[i,j]` to `v`, so that `A[i,j]` reads back `v` (the
`AbstractArray` round-trip). The write is absorbed into the **sparse part** by storing
`S[i,j] = v - (U*V)[i,j]`; the low-rank correction `U*V` is left unchanged. To edit the
low-rank part instead, mutate `fillpart(A)` / `lowrankfactors(A)` directly.

Note this can introduce a stored entry into `S` at `(i,j)` (as any sparse `setindex!`
does), and on a low-rank-covered row stores the value minus the correction.
"""
Base.@propagate_inbounds function Base.setindex!(A::SparseWithDenseRowColMatrix, v, i::Int, j::Int)
    @boundscheck checkbounds(A, i, j)
    @inbounds A.S[i, j] = v - _lowrank_entry(A.U, A.V, i, j)
    return v
end

function Base.similar(A::SparseWithDenseRowColMatrix, ::Type{T}) where {T}
    return SparseWithDenseRowColMatrix{T}(similar(A.S, T), _similar_U(A.U, T), similar(A.V, T))
end
_similar_U(U::SelectorMatrix, ::Type{T}) where {T} = SelectorMatrix{T}(U.n, U.r)
_similar_U(U, ::Type{T}) where {T} = similar(U, T)

function Base.copy(A::SparseWithDenseRowColMatrix)
    return SparseWithDenseRowColMatrix(copy(A.S), _copy_U(A.U), copy(A.V))
end
_copy_U(U::SelectorMatrix{T}) where {T} = SelectorMatrix{T}(U.n, U.r)
_copy_U(U) = copy(U)

function Base.fill!(A::SparseWithDenseRowColMatrix, x)
    fill!(SparseArrays.nonzeros(A.S), x)  # structural nonzeros only — keeps the pattern
    A.U isa SelectorMatrix || fill!(A.U, x)
    fill!(A.V, x)
    return A
end

function Base.Matrix(A::SparseWithDenseRowColMatrix{T}) where {T}
    M = Matrix(A.S)
    mul!(M, _densify(A.U), A.V, one(T), one(T))
    return M
end
_densify(U::AbstractMatrix) = U
_densify(U::SelectorMatrix{T}) where {T} = materialize_U!(Matrix{T}(undef, U.n, U.r), U)

# ------------------
# Pretty printing
# ------------------

function Base.array_summary(io::IO, A::SparseWithDenseRowColMatrix{T}, inds::Tuple{Vararg{Base.OneTo}}) where {T}
    print(
        io, Base.dims2string(length.(inds)), " SparseWithDenseRowColMatrix{$T} with ",
        SparseArrays.nnz(A.S), " stored entries and fill rank ", denserank(A)
    )
    return
end
