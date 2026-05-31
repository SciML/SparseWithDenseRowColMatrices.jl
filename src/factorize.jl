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

"""
    qr(A::SparseWithDenseRowColMatrix; strategy=:auto) -> SparseWithDenseRowColQRAugmented

QR-factorize `A = S + U V` for numerically stable repeated solves. Builds a single
rank-revealing, column-pivoted sparse QR ([SparseColumnPivotedQR](https://github.com/SciML/SparseColumnPivotedQR.jl))
of the bordered system `[S U; V -I]`, returning a [`SparseWithDenseRowColQRAugmented`](@ref).
Unlike [`factorize`](@ref)/[`lu`](@ref) (the LU/Woodbury throughput path) this never forms
`S⁻¹`, so it stays accurate even when the sparse part `S` is ill-conditioned or nearly singular,
as long as `A` itself is nonsingular — the FastAlmostBandedMatrices regime.

* `strategy = :auto` (default) or `:augmented`: the augmented bordered-system QR (the only mode).
* `strategy = :woodbury`: **not supported** — a Woodbury-over-qr(S) approach shares the
  κ(S)·κ(C) cancellation of the LU Woodbury path and is catastrophically inaccurate on
  ill-conditioned `S`, defeating the purpose of using QR. Throws.

The solve is **allocation-free** and [`refactor!`](@ref) / [`qr!`](@ref) **reuses the symbolic
analysis** (column ordering / elimination tree) for a fixed sparsity pattern, so the QR path is
usable in a Newton / time-stepping hot loop, not only for one-off stable solves. Any element
type the backend supports works (the BLAS floats plus generic numbers such as `BigFloat` and
`ForwardDiff.Dual`). Solve with `F \\ b` / `ldiv!(F, b)`; update values with `refactor!`/`qr!`.
"""
function LinearAlgebra.qr(A::SparseWithDenseRowColMatrix; strategy::Symbol = :auto)
    strategy in (:auto, :augmented) && return _augmented_qr(A)
    strategy === :woodbury && throw(
        ArgumentError(
            "qr does not support strategy = :woodbury: a Woodbury-over-qr(S) solve shares the \
         κ(S)·κ(C) cancellation of the LU Woodbury path and is inaccurate on ill-conditioned S, \
         defeating the purpose of QR. Use `qr(A; strategy = :augmented)` (the default), or \
         `factorize(A; strategy = :woodbury)` for the LU Woodbury path."
        )
    )
    throw(ArgumentError("strategy must be :auto or :augmented for qr; got :$strategy"))
end

"""
    qr!(F::SparseWithDenseRowColQRAugmented, A::SparseWithDenseRowColMatrix; kwargs...) -> F

Alias for [`refactor!`](@ref) — re-`qr` `F` in place with the new values of `A`, reusing the
cached symbolic analysis (column ordering / elimination tree) for a fixed sparsity pattern.
"""
LinearAlgebra.qr!(F::SparseWithDenseRowColQRAugmented, A::SparseWithDenseRowColMatrix; kwargs...) =
    refactor!(F, A; kwargs...)

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

"""
    SparseWithDenseRowColQRFactorization(; check_pattern=true)

A [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) algorithm that solves
`SparseWithDenseRowColMatrix` systems with the numerically stable augmented sparse QR
([`SparseWithDenseRowColQRAugmented`](@ref)) through LinearSolve's caching interface. Opt-in
(the default LinearSolve algorithm for the matrix type remains the LU
[`SparseWithDenseRowColFactorization`](@ref)); choose this when `S` is ill-conditioned.

* `check_pattern` — validate the pattern on each refactor (`false` skips the check).

A value update reuses the QR symbolic analysis (`refactor!` calls the backend's `csr_refactor!`),
so there is no `reuse_symbolic` knob to set. There is no `strategy` (only the augmented mode
exists) and no `refine` (the augmented QR needs no Woodbury refinement). Only available when
`LinearSolve` is loaded.
"""
function SparseWithDenseRowColQRFactorization end
