using SparseWithDenseRowColMatrices
using Test

@testset "public API documentation" begin
    public_names = filter(
        !=(:SparseWithDenseRowColMatrices),
        names(SparseWithDenseRowColMatrices; all = false, imported = false),
    )

    missing_docs = Symbol[]
    for name in public_names
        binding = Docs.Binding(SparseWithDenseRowColMatrices, name)
        Docs.hasdoc(binding) || push!(missing_docs, name)
    end
    @test isempty(missing_docs)

    readme = read(joinpath(pkgdir(SparseWithDenseRowColMatrices), "README.md"), String)
    missing_readme_entries = filter(name -> !occursin(String(name), readme), public_names)
    @test isempty(missing_readme_entries)
end
