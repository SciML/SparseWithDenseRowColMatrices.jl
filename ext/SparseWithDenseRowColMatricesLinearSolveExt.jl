module SparseWithDenseRowColMatricesLinearSolveExt

using SparseWithDenseRowColMatrices
using SparseWithDenseRowColMatrices: SparseWithDenseRowColMatrix, SparseWithDenseRowColWoodbury,
    SparseWithDenseRowColAugmented, SparseWithDenseRowColQRAugmented, denserank, refactor!,
    SparseWithDenseRowColFactorization, SparseWithDenseRowColQRFactorization
using LinearSolve
using LinearSolve: LinearCache, OperatorAssumptions, LinearVerbosity, AbstractSparseFactorization
using LinearAlgebra: LinearAlgebra, ldiv!, issuccess, Factorization, SingularException, factorize, qr

# LinearSolve `@reexport using SciMLBase`, so SciMLBase is reachable through it.
const SciMLBase = LinearSolve.SciMLBase

# ------------------
# Algorithm
# ------------------
# Concrete algorithm type lives in the extension; the user-facing constructor is the
# `SparseWithDenseRowColFactorization` function exported by the package (a stub there, with
# its method defined here so it errors helpfully when LinearSolve is not loaded).
struct SWDRCFactorizationAlg <: AbstractSparseFactorization
    reuse_symbolic::Bool
    check_pattern::Bool
    strategy::Symbol
    refine::Int
end

function SparseWithDenseRowColMatrices.SparseWithDenseRowColFactorization(;
        reuse_symbolic::Bool = true, check_pattern::Bool = true,
        strategy::Symbol = :auto, refine::Integer = 1
    )
    return SWDRCFactorizationAlg(reuse_symbolic, check_pattern, strategy, Int(refine))
end

# Default algorithm LinearSolve picks for our matrix type, so `solve(LinearProblem(A, b))`
# uses the caching path automatically (analogous to KLU being the default for sparse Float64).
LinearSolve.defaultalg(::SparseWithDenseRowColMatrix, b, ::OperatorAssumptions{Bool}) =
    SWDRCFactorizationAlg(true, true, :auto, 1)

# ------------------
# Cache
# ------------------
# Type-stable cacheval wrapper. The inner factorization can be a Woodbury *or* an Augmented
# one — and can even switch between them across refactorizations if the values make `S`
# (non)singular — so we box it behind a fixed wrapper type rather than letting the
# `LinearCache` field type change underneath us.
mutable struct SWDRCCacheval{T}
    fact::Union{Nothing, Factorization{T}}
end

function LinearSolve.init_cacheval(
        ::SWDRCFactorizationAlg, A::SparseWithDenseRowColMatrix{T}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    ) where {T}
    return SWDRCCacheval{T}(nothing)
end

# Refresh the cached factorization for the current `A`, reusing the cached symbolic analysis
# when possible. `refactor!` reuses PureKLU's symbolic analysis (no re-analysis); it throws if
# the pattern/rank changed or if the (Woodbury) `S` became singular — in those cases we fall
# back to a full `factorize` (which can also switch Woodbury ↔ augmented).
#
# A genuinely singular `A` makes `factorize` throw `SingularException`. LinearSolve's contract
# (cf. its KLU path, which uses `check=false`) is to return `ReturnCode.Infeasible` rather than
# throw, so we catch it and store `nothing`; `solve!`'s `issuccess`/`!== nothing` guard then
# maps that to `Infeasible`. (Direct `factorize`/`\` on the matrix still throws, as is idiomatic.)
#
# NOTE (sticky fallback): once a singular `S` forces the augmented path, a later nonsingular `S`
# keeps reusing the augmented factorization (its `refactor!` succeeds for any nonsingular `A`),
# so the cache does not switch back to the faster Woodbury path on its own. Re-`init` the cache
# to recover the Woodbury path if `S` becomes reliably nonsingular again.
function _refresh!(wrap::SWDRCCacheval, A::SparseWithDenseRowColMatrix, alg::SWDRCFactorizationAlg)
    f = wrap.fact
    if alg.reuse_symbolic && f isa Factorization &&
            size(f) == size(A) && denserank(f) == denserank(A)
        try
            refactor!(f, A; check = alg.check_pattern)   # reuse cached symbolic analysis
            return wrap
        catch e
            (e isa SingularException || e isa ArgumentError || e isa DimensionMismatch) || rethrow(e)
        end
    end
    try
        wrap.fact = factorize(A; strategy = alg.strategy, refine = alg.refine)
    catch e
        e isa SingularException || rethrow(e)
        wrap.fact = nothing                              # singular A → solve! returns Infeasible
    end
    return wrap
end

function SciMLBase.solve!(cache::LinearCache, alg::SWDRCFactorizationAlg; kwargs...)
    if cache.isfresh
        wrap = LinearSolve.@get_cacheval(cache, :SWDRCFactorizationAlg)
        _refresh!(wrap, cache.A, alg)
        cache.isfresh = false
    end
    F = LinearSolve.@get_cacheval(cache, :SWDRCFactorizationAlg).fact
    return if F !== nothing && issuccess(F)
        y = ldiv!(cache.u, F, cache.b)
        SciMLBase.build_linear_solution(alg, y, nothing, cache; retcode = SciMLBase.ReturnCode.Success)
    else
        SciMLBase.build_linear_solution(
            alg, cache.u, nothing, cache; retcode = SciMLBase.ReturnCode.Infeasible
        )
    end
end

# ------------------
# QR algorithm (numerically stable, opt-in)
# ------------------
# Mirrors SWDRCFactorizationAlg but factors via the augmented sparse QR. There is no
# `reuse_symbolic` knob (a value update reuses the QR symbolic analysis via `refactor!` →
# `csr_refactor!`), no `strategy` (only the augmented mode exists), and no `refine`.
struct SWDRCQRFactorizationAlg <: AbstractSparseFactorization
    check_pattern::Bool
end

function SparseWithDenseRowColMatrices.SparseWithDenseRowColQRFactorization(; check_pattern::Bool = true)
    return SWDRCQRFactorizationAlg(check_pattern)
end

function LinearSolve.init_cacheval(
        ::SWDRCQRFactorizationAlg, A::SparseWithDenseRowColMatrix{T}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    ) where {T}
    return SWDRCCacheval{T}(nothing)
end

# Refresh the cached QR factorization. `refactor!` re-factors in place, reusing the cached
# symbolic analysis (and buffers) for a fixed pattern; it throws on a singular `A` or a changed
# pattern, in which case we rebuild with `qr`. A genuinely singular `A` makes `qr` throw `SingularException`,
# which we map to a `nothing` cache → `Infeasible` (matching the LU algorithm's contract).
function _refresh_qr!(wrap::SWDRCCacheval, A::SparseWithDenseRowColMatrix, alg::SWDRCQRFactorizationAlg)
    f = wrap.fact
    if f isa SparseWithDenseRowColQRAugmented &&
            size(f) == size(A) && denserank(f) == denserank(A)
        try
            refactor!(f, A; check = alg.check_pattern)
            return wrap
        catch e
            (e isa SingularException || e isa ArgumentError || e isa DimensionMismatch) || rethrow(e)
        end
    end
    try
        wrap.fact = qr(A)
    catch e
        e isa SingularException || rethrow(e)
        wrap.fact = nothing
    end
    return wrap
end

function SciMLBase.solve!(cache::LinearCache, alg::SWDRCQRFactorizationAlg; kwargs...)
    if cache.isfresh
        wrap = LinearSolve.@get_cacheval(cache, :SWDRCQRFactorizationAlg)
        _refresh_qr!(wrap, cache.A, alg)
        cache.isfresh = false
    end
    F = LinearSolve.@get_cacheval(cache, :SWDRCQRFactorizationAlg).fact
    return if F !== nothing && issuccess(F)
        y = ldiv!(cache.u, F, cache.b)
        SciMLBase.build_linear_solution(alg, y, nothing, cache; retcode = SciMLBase.ReturnCode.Success)
    else
        SciMLBase.build_linear_solution(
            alg, cache.u, nothing, cache; retcode = SciMLBase.ReturnCode.Infeasible
        )
    end
end

end # module
