using SafeTestsets, Test

@testset "SparseWithDenseRowColMatrices" begin
    @safetestset "Constructors & accessors" begin
        include("test_matrix.jl")
    end
    @safetestset "Matrix–vector / matrix multiply" begin
        include("test_matvec.jl")
    end
    @safetestset "Woodbury factorization & solve" begin
        include("test_woodbury.jl")
    end
    @safetestset "Augmented fallback" begin
        include("test_augmented.jl")
    end
    @safetestset "Refactorization (Newton path)" begin
        include("test_refactor.jl")
    end
    @safetestset "Generic eltypes" begin
        include("test_generic_eltypes.jl")
    end
    @safetestset "Edge cases & allocations" begin
        include("test_edgecases.jl")
    end
    @safetestset "Appropriateness detector" begin
        include("test_detect.jl")
    end
    @safetestset "LinearSolve caching integration" begin
        include("test_linearsolve.jl")
    end
    @safetestset "Arrow / bordered matrices" begin
        include("test_arrow.jl")
    end
    @safetestset "Aqua quality assurance" begin
        using Aqua, SparseWithDenseRowColMatrices
        # `ambiguities = false` (as in FastAlmostBandedMatrices.jl): the only remaining
        # ambiguities are inert cross-products of our matrix types with LinearAlgebra's
        # structured matrices (`Diagonal`/`Bidiagonal`/`Tridiagonal`/`Triangular`) in
        # `*`/`mul!` — combinations that never arise in use, and where both candidate
        # methods compute the same product anyway.
        Aqua.test_all(SparseWithDenseRowColMatrices; ambiguities = false, piracies = true)
    end
end
