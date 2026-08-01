module SparseWithDenseRowColMatricesIterativeSolversExt

using SparseWithDenseRowColMatrices
using SparseWithDenseRowColMatrices: SparseWithDenseRowColMatrix, _check_lstsq_eltype
using IterativeSolvers: IterativeSolvers
using LinearAlgebra: LinearAlgebra

# Iterative engine for `lstsq(A, b; alg=:iterative)`. LSQR (default) / LSMR are driven entirely
# by A's structured matvec and adjoint (`mul!(y, A, x)` / `mul!(y, A', x)` defined in
# src/matvec.jl), so the dense `A = S + U·V` is never formed. Started from x0 = 0, both
# converge to the MINIMUM-NORM least-squares solution for rank-deficient / inconsistent systems
# (the min-norm guarantee is contingent on the zero start — the solver is never given a guess).
function SparseWithDenseRowColMatrices._lstsq_iterative(
        A::SparseWithDenseRowColMatrix, b::AbstractVector;
        solver::Symbol = :lsqr, atol::Real = 1.0e-12, btol::Real = 1.0e-12,
        maxiters::Integer = 20 * size(A, 1), λ::Real = 0, verbose::Bool = false,
        # accept and ignore the dense-only keyword so `alg` can be flipped without edits
        rcond = nothing,
    )
    n = size(A, 1)
    length(b) == n || throw(DimensionMismatch("A has $n columns, b has length $(length(b))"))
    _check_lstsq_eltype(promote_type(eltype(A), eltype(b)))   # reject non-BLAS-float up front
    λ ≥ 0 || throw(ArgumentError("λ (Tikhonov damping) must be ≥ 0; got $λ"))

    x, ch = if solver === :lsqr
        IterativeSolvers.lsqr(A, b; damp = λ, atol = atol, btol = btol, maxiter = maxiters, log = true)
    elseif solver === :lsmr
        IterativeSolvers.lsmr(A, b; λ = λ, atol = atol, btol = btol, maxiter = maxiters, log = true)
    else
        throw(ArgumentError("solver must be :lsqr or :lsmr; got :$solver"))
    end

    # IterativeSolvers' `isconverged` is unreliable here (lsqr reports success even when it
    # stops at the iteration cap with a wrong answer), so flag on budget exhaustion — which
    # cleanly separates converged from non-converged for both lsqr and lsmr — and report the
    # true least-squares optimality ‖Aᴴ(Ax−b)‖ (one cheap structured adjoint matvec; →0 at a
    # minimizer) rather than trusting the solver's own flag.
    capped = ch.iters ≥ maxiters
    verbose && @info "lstsq(alg=:iterative)" solver converged = !capped iters = ch.iters
    if capped
        aresid = LinearAlgebra.norm(A' * (A * x .- b))
        @warn(
            "lstsq(...; alg=:iterative, solver=:$solver) hit the $maxiters-iteration budget " *
                "without converging — the result is NOT a reliable least-squares solution. " *
                "Raise `maxiters`, or use the exact `alg=:dense` if `A` is small enough to densify.",
            least_squares_residual = aresid,
            iters = ch.iters,
        )
    end
    return x
end

end # module
