module SparseWithDenseRowColMatrices

using LinearAlgebra
using LinearAlgebra: Factorization, SingularException, ldiv!, lu, lu!, qr, qr!, mul!, opnorm,
    issuccess, factorize
using SparseArrays
using SparseArrays: AbstractSparseMatrixCSC, SparseMatrixCSC, nonzeros, rowvals, getcolptr,
    nnz, sparse
using PureKLU
using SparseColumnPivotedQR
using SparseMatricesCSR: SparseMatrixCSR

import PrecompileTools: @setup_workload, @compile_workload

include("SelectorMatrix.jl")
include("matrix.jl")
include("matvec.jl")
include("woodbury.jl")
include("augmented.jl")
include("qr.jl")
include("factorize.jl")
include("lstsq.jl")
include("detect.jl")

export SparseWithDenseRowColMatrix, SelectorMatrix,
    sparsepart, fillpart, lowrankfactors, exclusive_sparsepart, denserank,
    SparseWithDenseRowColWoodbury, SparseWithDenseRowColAugmented,
    SparseWithDenseRowColQRAugmented, refactor!, update_lowrank!, lstsq,
    SparseWithDenseRowColLeastSquares,
    recommend_lowrank_peel, PeelRecommendation,
    SparseWithDenseRowColFactorization, SparseWithDenseRowColQRFactorization

# ---------------------
# Precompilation
# ---------------------

@setup_workload begin
    n, r = 12, 2
    for T in (Float64, ComplexF64)
        S = sparse(T(2) * I, n, n)
        @inbounds for i in 1:(n - 1)
            S[i + 1, i] = -one(T)
        end
        fill = ones(T, r, n)
        @compile_workload begin
            A = SparseWithDenseRowColMatrix(S, fill)
            b = ones(T, n)
            A * b
            F = factorize(A)
            F \ b
            refactor!(F, nonzeros(sparsepart(A)))
        end
    end
end

end # module
