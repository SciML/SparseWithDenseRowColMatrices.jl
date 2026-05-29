using SparseWithDenseRowColMatrices, SparseArrays, LinearAlgebra, Random

# Dense oracle for an SparseWithDenseRowColMatrix.
densify(A::SparseWithDenseRowColMatrix) = Matrix(A)

relerr(x, y) = norm(x .- y) / norm(y)

# A diagonally-dominant (well-conditioned) sparse bulk with some off-band sparsity, plus an
# r×n low-rank correction. `selector=true` uses the boundary-condition SelectorMatrix for U.
function rand_sparsedense(n, r; seed = 1, T = Float64, selector = false, scale = 0.1)
    rng = MersenneTwister(seed)
    S = sparse(T(4) * I, n, n)
    @inbounds for i in 1:(n - 1)
        S[i + 1, i] = T(-1)
        S[i, i + 1] = T(-1)
    end
    for _ in 1:(2n)            # a few off-band entries → genuinely sparse, not banded
        i = rand(rng, 1:n)
        j = rand(rng, 1:n)
        S[i, j] += T(rand(rng) * scale)
    end
    U = selector ? SelectorMatrix{T}(n, r) : T.(randn(rng, n, r))
    V = T.(randn(rng, r, n)) .* T(scale)
    return SparseWithDenseRowColMatrix(S, U, V)
end
