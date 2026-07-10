using SciMLTesting, SparseWithDenseRowColMatrices

include("public_api_docs.jl")

const LA = SparseWithDenseRowColMatrices.LinearAlgebra

# Non-public names accessed qualified from other packages / Base. Each goes public
# as its owning library declares it `public` (Julia >= 1.11), at which point the
# corresponding entry can be dropped:
#   Base:           @propagate_inbounds, OneTo, array_summary, dims2string
#   SparseArrays:   AbstractSparseMatrixCSC, getcolptr
#   LinearAlgebra:  AdjointFactorization, TransposeFactorization, gelsy! (LAPACK)
#   PureKLU:        KLUITypes
const QA_NONPUBLIC = (
    Symbol("@propagate_inbounds"), :OneTo, :array_summary, :dims2string,
    :AbstractSparseMatrixCSC, :getcolptr,
    :AdjointFactorization, :TransposeFactorization, :gelsy!,
    :KLUITypes,
)

run_qa(
    SparseWithDenseRowColMatrices;
    explicit_imports = true,
    # `ambiguities = false`: the only remaining ambiguities are inert cross-products of
    # our matrix types with LinearAlgebra's structured matrices (Diagonal/Bidiagonal/
    # Tridiagonal/Triangular) in `*`/`mul!` — combinations that never arise in use, and
    # where both candidate methods compute the same product anyway.
    # `persistent_tasks = (; tmax = 60)`: neither this package nor PureKLU spawns any task
    # at module load (no `@async`/`Threads.@spawn`/`Timer`/`__init__`; load exits in ~0.4s),
    # so the check has nothing real to catch. The default 10s budget for the spawned
    # subprocess to *exit* is flaky on a loaded CI host (heavier first-load on 1.12); the
    # larger budget removes that flake while still flagging a genuinely hung load.
    aqua_kwargs = (; ambiguities = false, persistent_tasks = (; tmax = 60)),
    ei_kwargs = (;
        # `tr` is a struct field / constructor parameter name in src/woodbury.jl, not the
        # LinearAlgebra `tr` function; ExplicitImports' static analysis can't tell the bare
        # local binding from the implicitly-available export, so ignore the false positive.
        no_implicit_imports = (; ignore = (:tr => LA,)),
        # `Adjoint`/`Transpose` are imported so the `@eval LinearAlgebra.mul!(..., ::$Wrap{...})`
        # dispatch methods in src/matvec.jl resolve them; the use is inside a quoted `@eval`,
        # which ExplicitImports cannot see, so it reports them as stale. Dropping them breaks
        # the adjoint/transpose matvec dispatch.
        no_stale_explicit_imports = (; ignore = (:Adjoint, :Transpose)),
        all_qualified_accesses_are_public = (; ignore = QA_NONPUBLIC),
    ),
)
