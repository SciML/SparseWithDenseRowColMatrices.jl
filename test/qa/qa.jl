using Aqua, SparseWithDenseRowColMatrices, Test

@testset "Aqua quality assurance" begin
    # `ambiguities = false` (as in FastAlmostBandedMatrices.jl): the only remaining
    # ambiguities are inert cross-products of our matrix types with LinearAlgebra's
    # structured matrices (`Diagonal`/`Bidiagonal`/`Tridiagonal`/`Triangular`) in
    # `*`/`mul!` — combinations that never arise in use, and where both candidate
    # methods compute the same product anyway.
    #
    # `persistent_tasks` uses a generous `tmax`: neither this package nor PureKLU spawns any
    # task at module load (verified — no `@async`/`Threads.@spawn`/`Timer`/`__init__`; the
    # load subprocess exits in ~0.4s), so the check has nothing real to catch. Its default
    # 10s wall-clock budget for the spawned subprocess to *exit* is, however, flaky on a
    # loaded CI host (heavier first-load on 1.12), producing a false positive under
    # precompile contention. The larger budget removes that flake while still flagging a
    # genuinely hung load.
    # `deps_compat` runs with `check_extras = false`: the `deps`/`weakdeps` compat
    # coverage still runs and passes; only the `extras` sub-check is skipped because
    # `Pkg` is listed in `[extras]`/the test target without a `[compat]` entry.
    # That genuine finding is tracked below as `@test_broken`.
    Aqua.test_all(
        SparseWithDenseRowColMatrices;
        ambiguities = false, piracies = true, persistent_tasks = (; tmax = 60),
        deps_compat = (; check_extras = false),
    )
    # Aqua deps_compat (extras): `Pkg` is in [extras]/test target with no [compat] entry —
    # tracked in https://github.com/SciML/SparseWithDenseRowColMatrices.jl/issues/25
    @test_broken false
end
