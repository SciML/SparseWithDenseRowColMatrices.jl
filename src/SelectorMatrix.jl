# ------------------
# SelectorMatrix
# ------------------

"""
    SelectorMatrix{T}(n, r)

The `n × r` matrix `[I_r; 0]` — i.e. its first `r` rows are the `r × r` identity and the
remaining `n - r` rows are zero. Used as the left low-rank factor `U` for the common
"dense rows live at the top" case of an [`SparseWithDenseRowColMatrix`](@ref): then `U * V` places
the `r × n` dense `fill` matrix `V` into the top `r` rows.

Stored implicitly (two `Int`s), so the boundary-condition case never materializes an
`n × r` dense `U` and the matrix–vector / factorization paths stay allocation-light.
"""
struct SelectorMatrix{T} <: AbstractMatrix{T}
    n::Int
    r::Int
    function SelectorMatrix{T}(n::Integer, r::Integer) where {T}
        r ≤ n || throw(ArgumentError("SelectorMatrix needs r ≤ n, got r=$r, n=$n"))
        return new{T}(Int(n), Int(r))
    end
end
SelectorMatrix(n::Integer, r::Integer) = SelectorMatrix{Bool}(n, r)

Base.size(M::SelectorMatrix) = (M.n, M.r)

@inline function Base.getindex(M::SelectorMatrix{T}, i::Int, j::Int) where {T}
    @boundscheck checkbounds(M, i, j)
    return (i == j && j ≤ M.r) ? one(T) : zero(T)
end

Base.IndexStyle(::Type{<:SelectorMatrix}) = IndexCartesian()

# A selector of one eltype is trivially convertible to another; used during promotion.
Base.convert(::Type{SelectorMatrix{T}}, M::SelectorMatrix) where {T} = SelectorMatrix{T}(M.n, M.r)
SelectorMatrix{T}(M::SelectorMatrix) where {T} = SelectorMatrix{T}(M.n, M.r)

"""
    materialize_U!(Z, U)

Copy the left low-rank factor `U` into the preallocated dense buffer `Z` (`n × r`). This is
the one place `U` is realized densely; specialized for [`SelectorMatrix`](@ref) so the
common case writes a sparse identity pattern instead of going through generic `getindex`.
"""
function materialize_U!(Z::AbstractMatrix, U::AbstractMatrix)
    copyto!(Z, U)
    return Z
end
function materialize_U!(Z::AbstractMatrix{T}, U::SelectorMatrix) where {T}
    fill!(Z, zero(T))
    @inbounds for i in 1:U.r
        Z[i, i] = one(T)
    end
    return Z
end
