module SparseWithDenseRowColMatrices

using LinearAlgebra: LinearAlgebra, SingularException, ldiv!, lu!, qr, mul!, opnorm, factorize,
    ColumnNorm, Diagonal, I, diag, norm, svd, Adjoint, Transpose
using SparseArrays: SparseArrays, SparseMatrixCSC, nonzeros, sparse
using PureKLU: PureKLU
using SparseColumnPivotedQR: SparseColumnPivotedQR

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
