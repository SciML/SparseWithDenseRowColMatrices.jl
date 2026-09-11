using SparseWithDenseRowColMatrices, BenchmarkTools
using StableRNGs, SparseArrays, LinearAlgebra

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

# Arrow-head style matrix: sparse diagonal + dense last row/col
n = 1000
S = sparse(I, n, n) .* 3.0
U = randn(rng, n, 1)
V = randn(rng, 1, n)
A = SparseWithDenseRowColMatrix(S, U, V)

x = rand(rng, n)
y = zeros(n)
b = rand(rng, n)

# =============================================================================
# Construction
# =============================================================================

SUITE["construct"] = BenchmarkGroup()

SUITE["construct"]["arrow"] = @benchmarkable SparseWithDenseRowColMatrix($S, $U, $V)
SUITE["construct"]["replace"] = @benchmarkable SparseWithDenseRowColMatrix(
    $S, $V; replace = true
)

# =============================================================================
# Operations
# =============================================================================

SUITE["ops"] = BenchmarkGroup()

SUITE["ops"]["matvec"] = @benchmarkable $A * $x
SUITE["ops"]["mul!"] = @benchmarkable mul!($y, $A, $x)
SUITE["ops"]["getindex"] = @benchmarkable $A[500, 500]
SUITE["ops"]["ldiv"] = @benchmarkable $A \ $b
