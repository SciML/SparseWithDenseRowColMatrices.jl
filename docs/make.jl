using Documenter
using LinearAlgebra
using SparseWithDenseRowColMatrices

makedocs(;
    modules = [SparseWithDenseRowColMatrices],
    sitename = "SparseWithDenseRowColMatrices.jl",
    pages = [
        "Home" => "index.md",
        "Public API" => "api.md",
    ],
    checkdocs = :exports,
)
