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
# Two engines:
#   :dense (default)     — complete-orthogonal decomposition (LAPACK gelsy) of the densified A.
#                          Exact (matches `pinv(Matrix(A))*b`), O(n³) time / O(n²) memory. Since
#                          A = S + U·V is always fully dense (the rank-r outer product fills
#                          every entry), there is no sparsity to exploit in a direct solve.
#   :iterative           — LSQR/LSMR (IterativeSolvers) driven by A's structured matvec/adjoint,
#                          never forming the dense A. For n too large to densify. Approximate to
#                          a tolerance, fragile under ill-conditioning. Lives in an extension
#                          (`using IterativeSolvers`).

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
    lstsq(A::SparseWithDenseRowColMatrix, b::AbstractVector; alg=:dense, kwargs...) -> x

Minimum-norm least-squares solution `x ≈ A⁺b` (the Moore–Penrose pseudoinverse solution),
correct for **rank-deficient and/or inconsistent** systems where `\\`/`factorize`/`qr` would
throw. Among all `x` minimizing `‖A x − b‖₂`, returns the one of smallest `‖x‖₂`.

* `alg = :dense` (default): complete-orthogonal decomposition (LAPACK `gelsy`) of the densified
  `A`. Exact, `O(n³)` time and `O(n²)` memory. `A = S + U·V` is always fully dense, so a direct
  solve cannot exploit the sparsity; this is the right choice up to `n` of a few thousand.
* `alg = :iterative`: LSQR (default) / LSMR driven by `A`'s structured matvec and adjoint,
  never forming the dense `A` — for `n` too large to densify. Requires `using IterativeSolvers`.
  Approximate to a tolerance and fragile under ill-conditioning; see the keyword options.

Keywords: `rcond` (`:dense` rank cutoff), `solver` (`:lsqr`/`:lsmr`), `atol`, `btol`,
`maxiters`, `λ` (Tikhonov damping; `0` = pure minimum-norm), `verbose` (`:iterative`).
Only `Float32`/`Float64`/`ComplexF32`/`ComplexF64` are supported.

`\\`, `factorize`/`lu`, and `qr` are unchanged — they stay exact and throw on singular `A`;
least squares is always an explicit opt-in via `lstsq`.
"""
function lstsq(A::SparseWithDenseRowColMatrix, b::AbstractVector; alg::Symbol = :dense, kwargs...)
    if alg === :dense
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
        throw(ArgumentError("alg must be :dense or :iterative; got :$alg"))
    end
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
