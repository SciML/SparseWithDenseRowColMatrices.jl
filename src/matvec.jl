# ------------------
# SelectorMatrix products  ( U = [I_r; 0] )
# ------------------
# U*w places w (length r / r rows) into the top r entries/rows and zeros the rest.
# Split into vector / matrix methods so each is strictly more specific than the generic
# LinearAlgebra `mul!` (which would otherwise be ambiguous on the SelectorMatrix argument).

function _selector_mul!(y, U::SelectorMatrix, w, α::Number, β::Number)
    size(w, 1) == U.r ||
        throw(DimensionMismatch("SelectorMatrix has $(U.r) columns, operand has $(size(w, 1)) rows"))
    size(y, 1) == U.n ||
        throw(DimensionMismatch("output has $(size(y, 1)) rows, expected $(U.n)"))
    r, n = U.r, U.n
    if iszero(β)
        @inbounds for c in axes(w, 2)
            for i in 1:r
                y[i, c] = α * w[i, c]
            end
            for i in (r + 1):n
                y[i, c] = zero(eltype(y))
            end
        end
    else
        @inbounds for c in axes(w, 2)
            for i in 1:r
                y[i, c] = α * w[i, c] + β * y[i, c]
            end
            for i in (r + 1):n
                y[i, c] = β * y[i, c]
            end
        end
    end
    return y
end

# Only the *vector* method is registered as `LinearAlgebra.mul!` (it is strictly more
# specific than the generic matvec and so unambiguous, and it is what the polymorphic
# `_applyA!` calls when `U` may be a selector). The matrix case is reached only through the
# internal calls below, which dispatch to `_selector_mul!` directly — registering it as
# `mul!` would clash with LinearAlgebra's `mul!(::AbstractMatrix, ::AbstractMatrix, ::Diagonal/Triangular)`.
LinearAlgebra.mul!(y::AbstractVector, U::SelectorMatrix, w::AbstractVector, α::Number, β::Number) =
    _selector_mul!(y, U, w, α, β)

function Base.:*(U::SelectorMatrix{T}, w::AbstractVector) where {T}
    y = Vector{promote_type(T, eltype(w))}(undef, U.n)
    return _selector_mul!(y, U, w, true, false)
end
function Base.:*(U::SelectorMatrix{T}, W::AbstractMatrix) where {T}
    Y = Matrix{promote_type(T, eltype(W))}(undef, U.n, size(W, 2))
    return _selector_mul!(Y, U, W, true, false)
end

# ------------------
# SparseWithDenseRowColMatrix * vector / matrix:  A = S + U*V
# ------------------

# `y .+= c * U[:, k]` — alloc-free for both the selector and dense cases.
@inline function _axpy_col!(y::AbstractVector, U::SelectorMatrix, k::Int, c)
    @inbounds y[k] += c
    return y
end
@inline function _axpy_col!(y::AbstractVector, U::AbstractMatrix, k::Int, c)
    @inbounds for i in eachindex(y)
        y[i] += c * U[i, k]
    end
    return y
end

"""
    mul!(y, A::SparseWithDenseRowColMatrix, x, α, β)

Five-argument multiply `y .= α*(A*x) + β*y`. The vector path is allocation-free: the
sparse `S*x` uses `SparseArrays`' in-place kernel and the rank-`r` correction is applied as
`r` scaled column updates without forming the intermediate `V*x`.
"""
function LinearAlgebra.mul!(
        y::AbstractVector, A::SparseWithDenseRowColMatrix, x::AbstractVector, α::Number, β::Number
    )
    mul!(y, A.S, x, α, β)
    U, V = A.U, A.V
    @inbounds for k in axes(V, 1)
        sk = zero(promote_type(eltype(V), eltype(x)))
        for j in eachindex(x)
            sk += V[k, j] * x[j]
        end
        _axpy_col!(y, U, k, α * sk)
    end
    return y
end
LinearAlgebra.mul!(y::AbstractVector, A::SparseWithDenseRowColMatrix, x::AbstractVector) =
    mul!(y, A, x, true, false)

function Base.:*(A::SparseWithDenseRowColMatrix, x::AbstractVector)
    T = promote_type(eltype(A), eltype(x))
    y = Vector{T}(undef, size(A, 1))
    return mul!(y, A, x, one(T), zero(T))
end

# Matrix RHS: forming the r×ncols block V*X is cheap (r ≪ n) and not the solve hot path.
function LinearAlgebra.mul!(
        Y::AbstractMatrix, A::SparseWithDenseRowColMatrix, X::AbstractMatrix, α::Number, β::Number
    )
    mul!(Y, A.S, X, α, β)
    W = A.V * X                       # r × ncols
    _add_lowrank!(Y, A.U, W, α)       # Y .+= α * U*(V*X)
    return Y
end
_add_lowrank!(Y, U::SelectorMatrix, W, α) = _selector_mul!(Y, U, W, α, true)
_add_lowrank!(Y, U, W, α) = mul!(Y, U, W, α, true)
LinearAlgebra.mul!(Y::AbstractMatrix, A::SparseWithDenseRowColMatrix, X::AbstractMatrix) =
    mul!(Y, A, X, true, false)

function Base.:*(A::SparseWithDenseRowColMatrix, X::AbstractMatrix)
    T = promote_type(eltype(A), eltype(X))
    Y = Matrix{T}(undef, size(A, 1), size(X, 2))
    return mul!(Y, A, X, one(T), zero(T))
end

# ------------------
# Adjoint / transpose matvec:  Aᴴu = Sᴴu + Vᴴ(Uᴴu)   (and Aᵀ with no conjugation)
# ------------------
# Without these, `mul!(y, A', u)` / `A'*u` fall back to LinearAlgebra's generic `getindex`
# path, which materializes columns of `A` and is ~1000× slower than the structured product —
# crippling for an iterative least-squares solve, whose every step needs an adjoint matvec.

# w = Uᴴu (length r). Selector U = [I_r; 0] ⇒ (Uᴴu)_k = u_k.
function _lowrank_adjU!(w, U::AbstractMatrix, u, cnj)
    @inbounds for k in axes(U, 2)
        s = zero(eltype(w))
        for i in axes(U, 1)
            s += cnj(U[i, k]) * u[i]
        end
        w[k] = s
    end
    return w
end
function _lowrank_adjU!(w, U::SelectorMatrix, u, cnj)
    @inbounds for k in 1:U.r
        w[k] = u[k]
    end
    return w
end

# `Sop` is `adjoint`/`transpose`, `cnj` is `conj`/`identity` (for the dense U/V blocks).
function _adjoint_matvec!(
        y::AbstractVector, A::SparseWithDenseRowColMatrix, u::AbstractVector,
        α::Number, β::Number, Sop, cnj
    )
    mul!(y, Sop(A.S), u, α, β)                 # y .= α Sᴴu + β y
    U, V = A.U, A.V
    r = size(V, 1)
    w = Vector{promote_type(eltype(A), eltype(u))}(undef, r)
    _lowrank_adjU!(w, U, u, cnj)               # w = Uᴴu  (length r)
    @inbounds for k in 1:r
        wk = α * w[k]
        for j in axes(V, 2)
            y[j] += cnj(V[k, j]) * wk           # y .+= α Vᴴw
        end
    end
    return y
end

LinearAlgebra.mul!(
    y::AbstractVector, wA::Adjoint{<:Any, <:SparseWithDenseRowColMatrix}, u::AbstractVector,
    α::Number, β::Number,
) = _adjoint_matvec!(y, parent(wA), u, α, β, adjoint, conj)
LinearAlgebra.mul!(
    y::AbstractVector, wA::Adjoint{<:Any, <:SparseWithDenseRowColMatrix}, u::AbstractVector,
) = mul!(y, wA, u, true, false)

LinearAlgebra.mul!(
    y::AbstractVector, wA::Transpose{<:Any, <:SparseWithDenseRowColMatrix}, u::AbstractVector,
    α::Number, β::Number,
) = _adjoint_matvec!(y, parent(wA), u, α, β, transpose, identity)
LinearAlgebra.mul!(
    y::AbstractVector, wA::Transpose{<:Any, <:SparseWithDenseRowColMatrix}, u::AbstractVector,
) = mul!(y, wA, u, true, false)
