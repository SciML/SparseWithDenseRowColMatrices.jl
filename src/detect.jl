# ------------------
# Appropriateness detector
# ------------------

"""
    PeelRecommendation

The result of [`recommend_lowrank_peel`](@ref): whether wrapping a sparse matrix as a
[`SparseWithDenseRowColMatrix`](@ref) (and using the Woodbury path) is worthwhile, together
with the cheap pattern diagnostics that drove the decision. It carries **no eagerly-built
message string** — construction is allocation-free (only `Bool`/`Int`/`Float64`/`Symbol`
fields); the human-readable explanation is produced lazily by `show`.

# Fields
* `recommended :: Bool`   — use a `SparseWithDenseRowColMatrix` (`true`) or plain sparse LU (`false`)
* `rank :: Int`           — proposed low-rank correction size `r` (dense rows + dense cols)
* `code :: Symbol`        — reason category, one of `:recommended`, `:uniformly_sparse`,
  `:rank_too_large`, `:negligible_fill`, `:too_small`, `:not_square`, `:empty`
* `n :: Int`, `nnz :: Int`
* `dense_rows :: Int`, `dense_cols :: Int`
* `max_row_nnz :: Int`, `max_col_nnz :: Int`
* `row_threshold :: Float64`, `col_threshold :: Float64`
* `peeled_fraction :: Float64`   — share of the nonzeros held by the dense rows/cols

`Bool(rec)` and `rec.recommended` both give the verdict; `rec.rank` the proposed `r`.
"""
struct PeelRecommendation
    recommended::Bool
    rank::Int
    code::Symbol
    n::Int
    nnz::Int
    dense_rows::Int
    dense_cols::Int
    max_row_nnz::Int
    max_col_nnz::Int
    row_threshold::Float64
    col_threshold::Float64
    peeled_fraction::Float64
end

Base.Bool(rec::PeelRecommendation) = rec.recommended

# Lazily render the explanation from the stored diagnostics (no string built until shown).
function _peel_reason(rec::PeelRecommendation)
    c = rec.code
    if c === :not_square
        return "matrix is not square ($(rec.n) rows); the peel method targets square systems"
    elseif c === :too_small
        return "n=$(rec.n) is too small for a low-rank peel to matter; use plain sparse LU"
    elseif c === :empty
        return "matrix is structurally empty; use plain sparse LU"
    elseif c === :uniformly_sparse
        return "uniformly sparse: max $(rec.max_row_nnz) nnz/row, max $(rec.max_col_nnz) nnz/col " *
            "(thresholds $(round(rec.row_threshold; digits = 1)) / $(round(rec.col_threshold; digits = 1))); " *
            "no dense rows/cols — use plain sparse LU"
    elseif c === :rank_too_large
        return "found r=$(rec.rank) dense rows/cols ($(rec.dense_rows) rows + $(rec.dense_cols) cols); " *
            "the low-rank correction is too large relative to n=$(rec.n) — use plain sparse LU"
    elseif c === :negligible_fill
        return "found r=$(rec.rank) dense rows/cols but they hold only " *
            "$(round(100 * rec.peeled_fraction; digits = 1))% of the nonzeros; the bulk barely changes — use plain sparse LU"
    elseif c === :recommended
        return "found $(rec.dense_rows) dense row(s) (max $(rec.max_row_nnz) nnz) and $(rec.dense_cols) dense col(s) " *
            "(max $(rec.max_col_nnz) nnz); peeling rank r=$(rec.rank) removes ~$(round(100 * rec.peeled_fraction; digits = 1))% of the nnz, " *
            "leaving a sparse bulk → SparseWithDenseRowColMatrix recommended"
    else
        return "unknown"
    end
end

function Base.show(io::IO, rec::PeelRecommendation)
    return print(io, "PeelRecommendation(recommended=", rec.recommended, ", rank=", rec.rank, ", :", rec.code, ")")
end
function Base.show(io::IO, ::MIME"text/plain", rec::PeelRecommendation)
    println(
        io, "PeelRecommendation: ", rec.recommended ? "use SparseWithDenseRowColMatrix" : "use plain sparse LU",
        " (rank r=", rec.rank, ")"
    )
    return print(io, "  ", _peel_reason(rec))
end

"""
    recommend_lowrank_peel(A::AbstractSparseMatrixCSC;
                           row_factor=8.0, col_factor=8.0, maxrank=32,
                           min_n=64, abs_floor=4) -> PeelRecommendation

Cheap, pattern-only (`O(nnz)`, never factorizes) heuristic that decides whether wrapping `A`
as a [`SparseWithDenseRowColMatrix`](@ref) and solving with the Woodbury path is worth it,
versus just using plain sparse LU (e.g. `PureKLU.klu`). Returns a [`PeelRecommendation`](@ref)
(a descriptive struct — no allocating message string is built unless you `show` it).

The method pays off **iff** a small number `r ≪ n` of rows and/or columns are *much* denser
than the rest, so peeling them into the low-rank correction `U*V` leaves a genuinely sparse,
low-fill bulk `S`. On a uniformly sparse matrix the low-rank part is empty and the method
degenerates to plain LU — then `recommended` is `false`.

A row (column) is a dense candidate when its stored-entry count exceeds both
`row_factor * median(nnz per row)` (resp. `col_factor`) and the absolute floor `abs_floor`.
The recommendation also requires the candidates to be few (`r ≤ min(maxrank, n ÷ 8)`) and to
hold a meaningful share (≥ 5%) of the nonzeros.

# Example
```julia
julia> rec = recommend_lowrank_peel(A);

julia> rec.recommended, rec.rank
(false, 0)
```
"""
function recommend_lowrank_peel(
        A::SparseArrays.AbstractSparseMatrixCSC;
        row_factor::Real = 8.0, col_factor::Real = 8.0,
        maxrank::Int = 32, min_n::Int = 64, abs_floor::Int = 4
    )
    n, m = size(A)
    nnzA = SparseArrays.nnz(A)

    n == m || return PeelRecommendation(false, 0, :not_square, n, nnzA, 0, 0, 0, 0, 0.0, 0.0, 0.0)
    n < min_n && return PeelRecommendation(false, 0, :too_small, n, nnzA, 0, 0, 0, 0, 0.0, 0.0, 0.0)
    nnzA == 0 && return PeelRecommendation(false, 0, :empty, n, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0)

    colptr = SparseArrays.getcolptr(A)
    rows = SparseArrays.rowvals(A)
    colcnt = diff(colptr)                         # nnz per column, O(n)
    rowcnt = zeros(Int, n)
    @inbounds for k in 1:nnzA
        rowcnt[rows[k]] += 1
    end

    row_thresh = max(row_factor * max(_median_int(rowcnt), 1.0), float(abs_floor))
    col_thresh = max(col_factor * max(_median_int(colcnt), 1.0), float(abs_floor))
    dense_rows = count(>(row_thresh), rowcnt)
    dense_cols = count(>(col_thresh), colcnt)
    r = dense_rows + dense_cols
    maxrow, maxcol = maximum(rowcnt), maximum(colcnt)

    r == 0 && return PeelRecommendation(
        false, 0, :uniformly_sparse, n, nnzA, 0, 0, maxrow, maxcol, row_thresh, col_thresh, 0.0
    )

    rank_cap = min(maxrank, n ÷ 8)
    r > rank_cap && return PeelRecommendation(
        false, r, :rank_too_large, n, nnzA, dense_rows, dense_cols, maxrow, maxcol, row_thresh, col_thresh, 0.0
    )

    peeled = 0
    @inbounds for j in eachindex(colcnt)
        colcnt[j] > col_thresh && (peeled += colcnt[j])
    end
    @inbounds for i in eachindex(rowcnt)
        rowcnt[i] > row_thresh && (peeled += rowcnt[i])
    end
    frac = peeled / nnzA
    frac < 0.05 && return PeelRecommendation(
        false, r, :negligible_fill, n, nnzA, dense_rows, dense_cols, maxrow, maxcol, row_thresh, col_thresh, frac
    )

    return PeelRecommendation(
        true, r, :recommended, n, nnzA, dense_rows, dense_cols, maxrow, maxcol, row_thresh, col_thresh, frac
    )
end

# Median of a small integer count vector, without depending on Statistics.
function _median_int(v::AbstractVector{<:Integer})
    isempty(v) && return 0.0
    s = sort(v)
    L = length(s)
    return isodd(L) ? float(s[(L + 1) ÷ 2]) : (s[L ÷ 2] + s[L ÷ 2 + 1]) / 2
end
