using Pkg

const GROUP = get(ENV, "GROUP", "All")

if GROUP == "QA"
    Pkg.activate(joinpath(@__DIR__, "qa"))
    Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..")))
    Pkg.instantiate()
    include("qa.jl")
else
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
        @safetestset "QR factorization (augmented, stable)" begin
            include("test_qr.jl")
        end
        @safetestset "Rank-deficient least squares (min-norm A⁺b)" begin
            include("test_lstsq.jl")
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
    end
end
