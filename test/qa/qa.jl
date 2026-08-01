using SciMLTesting, SparseWithDenseRowColMatrices, Test

# ExplicitImports only walks an extension that is actually loaded (it resolves each
# `[extensions]` entry through `Base.get_extension`, which is `nothing` otherwise), so
# every weakdep is loaded here to bring the extension modules into the check.
using IterativeSolvers, LinearSolve

# ExplicitImports silently skips an extension that fails to load, so assert the
# extension modules actually exist rather than trusting a green run_qa.
@testset "Extensions loaded" begin
    exts = (
        :SparseWithDenseRowColMatricesIterativeSolversExt,
        :SparseWithDenseRowColMatricesLinearSolveExt,
    )
    for ext in exts
        @test Base.get_extension(SparseWithDenseRowColMatrices, ext) !== nothing
    end
end

run_qa(
    SparseWithDenseRowColMatrices;
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            ignore = (
                # SparseWithDenseRowColMatrices' own internals, reached from its own
                # extension. ExplicitImports treats an extension as a separate module,
                # so these read as non-public cross-module accesses even though they
                # never leave the package. `_lstsq_iterative` is the internal stub the
                # IterativeSolvers extension adds a method to.
                :_lstsq_iterative,
                # LinearSolve non-public: the algorithm-author interface. `defaultalg`
                # and `init_cacheval` are the hooks a new `AbstractFactorization` has to
                # define, and `@get_cacheval` is the only way to read the cacheval back
                # out in `solve!`. LinearSolve declares none of them public.
                :defaultalg, :init_cacheval, :var"@get_cacheval",
                # SciMLBase non-public: `build_linear_solution` is the required
                # constructor for the `LinearSolution` a `solve!` method must return.
                :build_linear_solution,
            ),
        ),
        all_explicit_imports_are_public = (;
            ignore = (
                # Package-internal eltype guard reused by the IterativeSolvers
                # extension; same cross-module artifact as `_lstsq_iterative` above.
                :_check_lstsq_eltype,
                # LinearSolve non-public: `LinearCache` is the cache type a `solve!`
                # method dispatches on and `AbstractSparseFactorization` is the
                # supertype a sparse algorithm must subtype. Neither is public.
                :AbstractSparseFactorization, :LinearCache,
            ),
        ),
    ),
)
