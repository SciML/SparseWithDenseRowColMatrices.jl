# ------------------
# Top-level factorize / lu / \  dispatch
# ------------------

const SparseWithDenseRowColFactorization{T} = Union{SparseWithDenseRowColWoodbury{T}, SparseWithDenseRowColAugmented{T}}

"""
    factorize(A::SparseWithDenseRowColMatrix; strategy=:auto, refine=1, auto_fallback=true) -> F

Factorize `A = S + U V` for repeated solves. Returns an [`SparseWithDenseRowColWoodbury`](@ref)
(the fast path: one sparse LU of `S` plus a small dense correction) or an
[`SparseWithDenseRowColAugmented`](@ref) (the robust bordered-system path).

* `strategy = :auto` (default): use Woodbury, automatically falling back to the augmented
  system if `S` is singular or the `r × r` correction is ill-conditioned.
* `strategy = :woodbury`: force Woodbury; warn on ill-conditioning, error on singular `S`.
* `strategy = :augmented`: force the bordered-system path.
* `refine`: iterative-refinement steps for the Woodbury solve (default `1`; ignored by the
  augmented path).
* `auto_fallback`: whether `:auto` may switch to the augmented path.

The returned `F` **caches the symbolic analysis** (BTF + AMD ordering of `S`). Solve any
number of right-hand sides with `F \\ b` / `ldiv!(F, b)`, and update the numeric values for a
new matrix of the **same sparsity pattern** with [`refactor!`](@ref) / [`lu!`](@ref) — that
reuses the cached symbolic analysis (no re-analysis), which is the ≈7× per-step win in a
Newton / time-stepping loop. Only call `factorize`/`lu` afresh when the pattern changes.
"""
function LinearAlgebra.factorize(
        A::SparseWithDenseRowColMatrix; strategy::Symbol = :auto, refine::Integer = 1,
        auto_fallback::Bool = true
    )
    strategy in (:auto, :woodbury, :augmented) ||
        throw(ArgumentError("strategy must be :auto, :woodbury or :augmented; got :$strategy"))

    strategy === :augmented && return _augmented(A)

    F = try
        _woodbury(A; refine)
    catch e
        if e isa SingularException && strategy === :auto && auto_fallback
            return _augmented(A)
        end
        rethrow(e)
    end

    if F.illconditioned
        if strategy === :auto && auto_fallback
            return _augmented(A)
        else
            @warn "SparseWithDenseRowColWoodbury: the r×r correction matrix C is ill-conditioned; the \
                   Woodbury solve may lose accuracy. Consider `strategy = :augmented`."
        end
    end
    return F
end

"""
    lu(A::SparseWithDenseRowColMatrix; kwargs...) -> F

Alias for [`factorize`](@ref). Named `lu` (not `qr`) because the sparse bulk is factored by
an LU-based solver (PureKLU); `qr` is intentionally not provided.
"""
LinearAlgebra.lu(A::SparseWithDenseRowColMatrix; kwargs...) = factorize(A; kwargs...)

"""
    lu!(F, A::SparseWithDenseRowColMatrix; kwargs...) -> F

Alias for [`refactor!`](@ref) — refactor `F` in place with the new values of `A`.
"""
LinearAlgebra.lu!(F::SparseWithDenseRowColFactorization, A::SparseWithDenseRowColMatrix; kwargs...) =
    refactor!(F, A; kwargs...)

function LinearAlgebra.qr(::SparseWithDenseRowColMatrix)
    throw(ArgumentError("SparseWithDenseRowColMatrix is solved by an LU-based factorization; use `factorize`/`lu`, not `qr`."))
end

Base.:\(A::SparseWithDenseRowColMatrix, b::AbstractVecOrMat) = factorize(A) \ b

"""
    SparseWithDenseRowColFactorization(; reuse_symbolic=true, check_pattern=true,
                                        strategy=:auto, refine=1)

A [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) algorithm that solves
`SparseWithDenseRowColMatrix` systems through LinearSolve's caching interface, mirroring its
`KLUFactorization`: the symbolic analysis is cached in the `LinearCache`, repeated `solve!`s
reuse it, and updating the matrix to new values of the **same sparsity pattern** triggers a
numeric refactorization (via [`refactor!`](@ref)) that reuses the cached symbolic analysis.
It is also the default algorithm `LinearSolve` picks for a `SparseWithDenseRowColMatrix`, so
`solve(LinearProblem(A, b))` uses it automatically.

* `reuse_symbolic` — reuse the cached symbolic analysis across value changes (set `false` if
  the sparsity pattern may change between solves).
* `check_pattern` — validate the pattern on each refactor (`false` skips the check; may error
  if the pattern actually changes).
* `strategy`, `refine` — forwarded to [`factorize`](@ref).

Only available when `LinearSolve` is loaded (provided by a package extension).
"""
function SparseWithDenseRowColFactorization end
