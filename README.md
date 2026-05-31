# SparseWithDenseRowColMatrices.jl

[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

A fast implementation of **"almost sparse" matrices** — a sparse bulk plus a low-rank dense
correction — in Julia. It is the sparse-matrix analogue of
[FastAlmostBandedMatrices.jl](https://github.com/SciML/FastAlmostBandedMatrices.jl): where
that package handles a *banded* bulk with a few dense boundary rows (as in spectral
collocation), this one handles a *general sparse* bulk with a low-rank correction (as in
finite-element / finite-volume boundary-value problems, DAE systems, and KKT/bordered
systems), built for use in [BoundaryValueDiffEq.jl](https://github.com/SciML/BoundaryValueDiffEq.jl).

The sparse part is factored with [PureKLU.jl](https://github.com/SciML/PureKLU.jl) — a
pure-Julia port of SuiteSparse's KLU (BTF + AMD + Gilbert–Peierls LU) — and the low-rank
part is handled by the Sherman–Morrison–Woodbury identity, so a solve costs **one sparse LU
solve plus a small `r × r` dense correction**. Because PureKLU is generic over
`Union{Real, Complex}`, the whole pipeline works for `Float64`/`ComplexF64`, `BigFloat`, and
`ForwardDiff.Dual` — i.e. you can autodiff straight through the solve.

## The structure

An `SparseWithDenseRowColMatrix` represents

```
A = S + U * V
```

* `S :: SparseMatrixCSC` — the `n × n` sparse bulk,
* `U` — `n × r` dense (or a `SelectorMatrix` `[I_r; 0]` for the boundary-row case),
* `V` — `r × n` dense, the low-rank "fill".

with `r ≪ n` the *almost-sparse rank*.

**Arrowhead / bordered matrices** are the canonical case: an arrowhead matrix (a sparse —
often diagonal — bulk plus a dense last row *and* last column) is exactly `S + U*V` with
`r = 2` (the border is rank-2), and a block-bordered system (sparse interior with a dense
border of width `k`, as in KKT / saddle-point / domain-decomposition problems) is `r = 2k`
over a block-diagonal bulk. Solving them this way avoids the fill-in a naive dense or
unordered-sparse LU suffers on the dense border.

## High-level example

```julia
using SparseWithDenseRowColMatrices, SparseArrays, LinearAlgebra

n, r = 1000, 4

# A diagonally-dominant sparse bulk
S = sparse(4.0I, n, n)
for i in 1:n-1; S[i+1, i] = -1.0; S[i, i+1] = -1.0; end

# A general low-rank correction A = S + U*V …
U = randn(n, r); V = randn(r, n)
A = SparseWithDenseRowColMatrix(S, U, V)

b = randn(n)
x = A \ b                       # factor once, solve — Woodbury over a PureKLU LU of S
F = factorize(A)                # keep the factorization for repeated solves
x = F \ b

# … or the boundary-condition form: `fill` is r dense rows placed at the top
fill = randn(r, n)
A2 = SparseWithDenseRowColMatrix(S, fill)               # adds `fill` to the top r rows
A3 = SparseWithDenseRowColMatrix(S, fill; replace=true) # top r rows *become* `fill`
```

### Newton / time-stepping: refactor in place

When the sparsity pattern is fixed and only the numeric values change (a Newton sweep on a
BVP, say), reuse the symbolic analysis and preallocated workspace:

```julia
F = factorize(A)
for step in 1:maxiters
    newvals = jacobian_values!(...)            # same pattern as S, new nzval
    refactor!(F, newvals)                      # reuses BTF/AMD analysis; no re-analysis
    Δ = F \ residual
    ...
end
```

The `refactor!` + `\` loop is allocation-free apart from the tiny `O(r)` dense pivot vector.

## Accessors (FastAlmostBandedMatrices parity)

| `AlmostBandedMatrix` | `SparseWithDenseRowColMatrix`           |
|----------------------|--------------------------------|
| `bandpart`           | `sparsepart`                   |
| `fillpart`           | `fillpart` (the `r × n` `V`)   |
| `almostbandedrank`   | `denserank`             |
| `exclusive_bandpart` | `exclusive_sparsepart`         |
| —                    | `lowrankfactors` → `(U, V)`    |

## Factorization strategy

`factorize(A; strategy=:auto, refine=1, auto_fallback=true)`:

* **`:woodbury`** (the fast path) — one PureKLU LU of `S`, precomputed `Z = S⁻¹U`, and a
  dense `r × r` correction `C = I + V Z`. Requires `S` and `C` nonsingular. The Woodbury
  forward error scales with `κ(S)·κ(C)`, so `refine` steps of (allocation-free) iterative
  refinement are applied (default `1`).
* **`:augmented`** (the robust path) — one sparse LU of the bordered system
  `[S U; V -I]` of size `n + r`. Requires only `A` (not `S`) to be nonsingular, so it is the
  correctness path when `S` is singular — common in BVPs where the interior stencil alone
  does not pin a boundary degree of freedom.
* **`:auto`** (default) — Woodbury, automatically falling back to the augmented system if
  `S` is singular or `C` is ill-conditioned.

### QR (the numerically stable path)

`qr(A; strategy=:augmented)` factors the bordered system `[S U; V -I]` with
[**SparseColumnPivotedQR**](https://github.com/SciML/SparseColumnPivotedQR.jl) (a pure-Julia,
rank-revealing, column-pivoted sparse QR — the QR analogue of PureKLU), returning a
`SparseWithDenseRowColQRAugmented`. Like the augmented LU path it never forms `S⁻¹`, so it stays
accurate when the sparse part `S` is ill-conditioned or nearly singular but the dense rows
regularize `A` — the
[FastAlmostBandedMatrices](https://github.com/SciML/FastAlmostBandedMatrices.jl) regime, where
QR is the conventional choice. Measured forward error on `A = S + U V` with `κ(A) ≈ 6` as
`κ(S)` sweeps from `3.6e6` to `3.6e14` (reference: BigFloat dense solve):

```
  κ(S)          qr (augmented)   factorize :woodbury, refine=2   factorize :augmented (LU)
  3.6e6   …   3.1e-16 (flat)        8.7e-17  →  4.5e-7              1.2e-16 (flat)
  3.6e14
```

The Woodbury path's error grows with `κ(S)·κ(C)`; here (where the *assembled* `A` stays
well-conditioned, `κ(A) ≈ 6`) both augmented paths hold at the noise floor.

**Be clear about what QR does and does not buy you.** Augmented QR is *not* more accurate than
the augmented **LU** path. It is backward-stable, so its forward error tracks `κ(A)·eps`; when
the *assembled* `A` is itself ill-conditioned the augmented-LU path's partial pivoting can be
**more** accurate on these structured matrices (measured: LU near the noise floor where QR sits
at `κ(A)·eps`, orders apart for `κ(A) ≳ 1e6`). QR's genuine, distinct properties versus
augmented-LU are that it is **rank-revealing** (a clean numerical-rank/singularity signal) and
that it is the QR-based interface users of almost-banded solvers expect — not higher accuracy,
which the LU path already delivers, and not throughput (it is slower; see the trade-offs
below). Reach for `qr` when you specifically want a QR factorization or its rank diagnostics;
otherwise `factorize`/`lu` is the better default.

There is **no** `strategy=:woodbury` for `qr`: a Woodbury-over-`qr(S)` solve shares the same
`κ(S)·κ(C)` cancellation and is catastrophically inaccurate on ill-conditioned `S` (~`1e-1`),
so it is deliberately not offered. Because the backend is genuinely rank-revealing (column
pivoting), `qr` throws `SingularException` exactly when the bordered system is rank-deficient;
a merely ill-conditioned, accurately-solvable `A` is **not** rejected.

The backend gives the QR path the same engineering niceties as the LU path: the solve is
**allocation-free**, [`refactor!`](@ref)/`qr!` **reuses the symbolic analysis** (column ordering
/ elimination tree) for a fixed sparsity pattern — so `qr` is usable in a Newton / time-stepping
hot loop, not only for one-off stable solves — and **any element type** works (the BLAS floats
plus generic numbers such as `BigFloat`/`ForwardDiff.Dual`, on every Julia version). Cost
(banded interior + `r` dense rows, factor / solve / refactor, ms):

```
  n=2000 r=4                          n=5000 r=8
  qr  (augmented QR) : 89 / 2.8 / 60    620 / 15 / 450    ← rank-revealing; alloc-free solve
  lu  (augmented LU) : 54 / 1.7 / 18    610 / 11 / 180    ← klu! symbolic reuse on refactor
  lu  (Woodbury)     : 1.3/ 0.1 / n/a   3.6 / 0.3 / n/a   ← fast path (forms S⁻¹), ~50–60×
```

QR is now in the same ballpark as the augmented-LU path (a bit slower on factor/refactor;
allocation-free solve), though still far from the Woodbury fast path. Forward error is
comparable (~`1e-13`) for all three on a well-conditioned `A`. Reach for `qr` when you want a
genuinely rank-revealing factorization, generic-eltype support, or the QR interface; `factorize`
/`lu` (Woodbury) remains the throughput default.

## Rank-deficient / inconsistent least squares (`lstsq`)

`\`, `factorize`/`lu`, and `qr` solve **nonsingular** systems and throw `SingularException`
when `A` is singular. For a genuinely rank-deficient or inconsistent system, `lstsq(A, b)`
returns the **minimum-norm least-squares solution** `x = A⁺b` (the Moore–Penrose solution: of
all `x` minimizing `‖Ax − b‖₂`, the one with smallest `‖x‖₂`).

```julia
x = lstsq(A, b)                        # :auto — structured direct, dense fallback (default)
x = lstsq(A, b; alg = :structured)     # structure-exploiting direct solve (never densifies A)
x = lstsq(A, b; alg = :dense)          # exact dense complete-orthogonal decomposition (gelsy)
x = lstsq(A, b; alg = :iterative)      # LSQR/LSMR (needs `using IterativeSolvers`)
x = lstsq(A, b; λ = 1e-3)              # Tikhonov-damped (ridge) variant
```

This is a **separate, opt-in** entry point — `\`/`factorize`/`qr` stay exact and keep throwing
on singular `A`; least squares never happens implicitly. The bordered/Woodbury machinery
*cannot* produce `A⁺b`: solving `[S U; V -I][x;y]=[b;0]` in least squares minimizes a different
objective than `‖Ax−b‖` (measured 3–65× too large a residual on rank-deficient inconsistent
systems), and there is no Woodbury-style formula for the pseudoinverse of a sum.

Engines (`:auto` picks the first that applies):

* **`:structured`** — a **direct** solve that **never densifies `A`**. When `S` is nonsingular,
  `A = S(I + ZV)` with `Z = S⁻¹U`, and the *entire* rank deficiency collapses into the small
  `r × r` capacitance `C = I + V Z` (`nullity(A) = nullity(C)`, since `det A = det S · det C`).
  So `A⁺b` is assembled from **one sparse factorization of `S`** (PureKLU) + `r` solves (`Z`) +
  small dense SVD/QR on `r × r` and `n × s` (`s ≤ 2r`) blocks — `O(factor(S) + n·r²)`, **>100×
  faster than `:dense`** for large `n` (measured 191× at `n = 2000`, `r = 6`). Handles
  consistent *and* inconsistent `b` (a rank-`k` projection of `b` onto `range(A)`). A **singular
  `S` with coordinate-aligned null structure** — the boundary-condition / `replace=true` case,
  where the zeroed rows are supplied by a `SelectorMatrix` `U` — is handled by a sparse *peel*
  `S̃ = S + U Uᴴ` (just `r` diagonal entries, so `S̃` stays sparse and is nonsingular) with
  `A = S̃ + U(V − Uᴴ)`, keeping the direct structure-exploiting path. It errors only when `S` is
  near-singular or singular with a *general* (dense) null space — where inverting `S` would be
  silently wrong — so `:auto` falls back to `:dense` there.
* **`:dense`** — complete-orthogonal decomposition (LAPACK `gelsy`) of the densified `A`. Exact
  (matches `pinv(Matrix(A))*b` to machine precision), `O(n³)` / `O(n²)`. The mandatory fallback
  when `S` is near-singular, or singular with a null space that is *not* coordinate-aligned (so
  the sparse peel does not apply and there is no structure-exploiting direct route — the
  null-space correction would itself be dense).
* **`:iterative`** — LSQR (default) / LSMR driven by `A`'s structured matvec/adjoint, never
  forming `A` — for `n` too large to densify *and* `S` singular (so neither `:structured` nor
  `:dense` apply). Started from `x0 = 0` it converges to the same minimum-norm solution;
  approximate (to `atol`/`btol`), fragile under ill-conditioning, and warns on non-convergence.
  Lives in an extension (`using IterativeSolvers`).

`:auto` (the default) uses `:structured` when `S` is nonsingular (or singular with a
coordinate-aligned null space) and well-conditioned, otherwise the exact `:dense` COD — so you
get the fast structure-exploiting direct solve when it is safe and the exact answer always. Only `Float32`/`Float64`/`ComplexF32`/`ComplexF64` are
supported. Adjoint/transpose matvec (`A'*u`, `Aᴴ`, `Aᵀ`) is computed by the same structured
kernel and is useful on its own.

For **repeated** least-squares solves with the same `A` (or a Newton / time-stepping loop),
build the factorization once and reuse it — the sparse factorization of `S` and the small dense
decompositions are cached, so each solve is a cheap back-solve:

```julia
F = SparseWithDenseRowColLeastSquares(A)   # structured setup once
x1 = F \ b1;  ldiv!(x2, F, b2);  …          # each solve reuses S's factorization
```

Measured (`n = 2000`, `r = 6`): a reused solve is `0.05 ms` / `16 KB` versus the one-shot
`lstsq(A, b)` at `2.1 ms` / `2.6 MB` — **~44× faster**, since it skips re-factoring `S`. (The
structured `alg=:auto`/`:structured` path applies; for a general singular `S` the constructor
errors and you use `lstsq(A, b; alg = :dense)`.)

## Benchmarks

A banded sparse interior plus `r = 8` dense rows (`n = 5000`). Factoring only `S` and
applying the rank-`r` Woodbury correction never assembles the dense rows into the
factorization, so it avoids the `O(n³)` dense cost and the fill those rows inflict on a
direct sparse `LU`. The honest baseline is **KLU** (PureKLU): its BTF + AMD ordering
*isolates* a few dense rows, so it handles the assembled matrix well — UMFPACK's COLAMD does
**not** (it fills), and dense `LU` is `O(n³)`.

```
n = 5000, r = 8   (factor + solve, solve rel err vs dense ≈ 1e-14)
  SWDRC (Woodbury)                 :   2.9 ms     ← this package
  KLU on the assembled S + U·V     :   4.9 ms     (1.7× slower; the right sparse baseline)
  UMFPACK lu on the assembled      : 345   ms     (~120×; COLAMD fills on the dense rows)
  dense lu                         : 820   ms     (~280×; O(n³))

  fixed S, changing low-rank correction — reuse, per solve:
  SWDRC  update_lowrank!           :   1.0 ms     ← reuses S's factorization
  KLU    klu! (refactor assembled) :   3.0 ms     (2.9× slower)
```

(`julia -O2`, single thread, `@elapsed` best-of. The win over **KLU** is modest at small
`r` (~1.5–3×) and grows with the border width `r` — KLU fills more as you add dense rows,
while the Woodbury correction stays `r×r`. The win over UMFPACK/dense is large because both
fill or scale as `O(n³)`. `S` must have good sparse structure for any of this to help; a
globally-scattered `S` degrades every sparse method.)

## What is fast

* [x] Matrix–vector / matrix–matrix multiply (`mul!` vector path is allocation-free)
* [x] Factorization and linear solve (`factorize`/`lu`, `\`, `ldiv!`)
* [x] In-place refactorization with a fixed sparsity pattern (`refactor!`)
* [x] Adjoint / transpose solves *and* structured adjoint/transpose matvec (`A'*u`, `Aᴴ`, `Aᵀ`)
* [x] Generic element types (`BigFloat`, `ForwardDiff.Dual`, complex) — LU/Woodbury *and* QR
* [x] Rank-revealing QR (`qr`) for ill-conditioned `S`: allocation-free solve, symbolic-reuse
  `refactor!`, any element type (column-pivoted, pure-Julia backend)
* [x] Minimum-norm least squares (`lstsq`) for rank-deficient / inconsistent `A` — a
  structure-exploiting **direct** solve (capacitance method) when `S` is nonsingular, >100×
  faster than densifying; exact dense COD fallback otherwise

`qr(A)` builds the augmented sparse-QR factorization (above): a rank-revealing column-pivoted
solve with an allocation-free `ldiv!` and a symbolic-reuse `refactor!`. It is comparable to the
augmented-LU path in cost; `factorize`/`lu` (Woodbury) is still the throughput default.

## When is this the right tool?

This package only beats plain sparse LU when the matrix is genuinely **sparse plus a few
dense rows/columns** — i.e. there exist `r ≪ n` rows/cols far denser than the rest, so
peeling them into `U·V` leaves a sparse, low-fill bulk `S`. On a *uniformly* sparse matrix
the low-rank part is empty (`r = 0`) and the method degenerates to plain `PureKLU.klu` with
no benefit. Use [`recommend_lowrank_peel`](@ref) — a cheap, pattern-only (`O(nnz)`, no
factorization) check — to decide:

```julia
rec = recommend_lowrank_peel(A)        # A::SparseMatrixCSC → a PeelRecommendation struct
rec.recommended   # false → just use plain sparse LU (PureKLU.klu) with symbolic reuse
                  # true  → wrap the rec.rank dense rows/cols as U·V and use this package
rec               # `show` renders the human-readable reason (built lazily, not on construction)
```

`recommend_lowrank_peel` returns a descriptive [`PeelRecommendation`](@ref) struct — only
`Bool`/`Int`/`Float64`/`Symbol` fields, so it costs nothing to construct; the explanation
string is formatted only when you display it.

### Cache the symbolic factorization, reuse it for many solves

For a fixed sparsity pattern with changing values (Newton / time-stepping), the win is
reusing the symbolic analysis (BTF + AMD ordering) instead of re-analyzing every step. The
factorization object caches it; `lu`/`factorize` analyzes once, `ldiv!`/`\` solves any
number of right-hand sides, and `lu!`/`refactor!` updates the values **reusing the cached
symbolic analysis** (no re-analysis):

```julia
F = lu(A)                     # analyze (cache symbolic) + numeric factor
x1 = F \ b1; x2 = F \ b2      # reuse for multiple solves
for step in 1:maxiters
    lu!(F, A_new)             # new values, same pattern → reuses cached symbolic analysis (no re-analysis)
    Δ = F \ residual
end
```

For a plain-sparse matrix (the `recommended == false` case) the same idea is `PureKLU.klu`
once + `klu!`/`solve!` per step.

### Fixed sparse base, changing low-rank correction — the Woodbury win

The structure pays off most in the classic Sherman–Morrison–Woodbury setting: a **fixed**
sparse base `S`, factored once, solved repeatedly under a **changing low-rank correction**
`U·V` — varying boundary conditions / coupling, adding or removing a constraint, sensitivity
or parameter sweeps, KKT / saddle-point systems with changing borders. Reusing `S`'s
factorization across the updates is the win: re-factoring the whole assembled `S + U·V`
each step redoes work that doesn't change.

[`update_lowrank!`](@ref) updates the low-rank factors while **reusing `S`'s cached
factorization** (no re-factorization of `S`, and `Z = S⁻¹U` is recomputed only if `U`
changes):

```julia
F = factorize(A)                       # factor S once
for step in 1:maxiters
    update_lowrank!(F; V = V_new)      # fixed S, new dense rows/coupling → reuse S's factorization
    x = F \ b
end
```

Measured (`n = 5000`, `r = 8`, `S` fixed, per solve):

```
  update_lowrank!  (reuse S's factorization)   :  1.0 ms / solve
  refactor!        (re-factors S each step)    :  ~3   ms / solve   (wastes the S re-factorization)
  KLU klu! on the refactored assembled S + U·V :  3.0 ms / solve   (~3× slower; grows with r)
```

The advantage over KLU is modest at small `r` and grows with the border width — KLU must
refactor the dense rows every step, while `update_lowrank!` touches only the `r×r`
correction. Use `update_lowrank!` when only the low-rank part changes, `refactor!` when
`S`'s values change (Newton), and `factorize` to (re)analyze a new pattern.

## LinearSolve.jl integration

Loading [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) activates a package
extension that plugs `SparseWithDenseRowColMatrix` into LinearSolve's caching interface
exactly like its `KLUFactorization`: the symbolic analysis is cached in the `LinearCache`,
repeated `solve!`s reuse the factorization, and assigning the matrix new values of the
**same sparsity pattern** triggers a numeric refactorization that reuses the cached symbolic
analysis. It is also the default algorithm for this matrix type.

```julia
using SparseWithDenseRowColMatrices, LinearSolve

prob  = LinearProblem(A, b)          # A::SparseWithDenseRowColMatrix
cache = init(prob)                   # default alg = SparseWithDenseRowColFactorization(); analyzes once
sol1  = solve!(cache)

cache.A = A_new                      # same pattern, new values → reuses cached symbolic (refactor!)
sol2  = solve!(cache)
cache.b = b_new                      # new RHS only → reuses the cached numeric factorization
sol3  = solve!(cache)
```

Pass `SparseWithDenseRowColFactorization(; reuse_symbolic, check_pattern, strategy, refine)`
explicitly to `init`/`solve` to control the behavior (mirrors `KLUFactorization(; …)`). If
`S` is singular it routes through the augmented fallback transparently, and `reuse_symbolic`
/ `check_pattern` behave as for KLU. A singular `A` returns `ReturnCode.Infeasible` (it does
not throw), again matching KLU. One caveat: once a singular `S` forces the augmented fallback,
the cache keeps reusing it even if later `S` values are nonsingular — `init` a fresh cache to
get back the faster Woodbury path.

For the stable QR path, pass `SparseWithDenseRowColQRFactorization(; check_pattern)` instead.
It has no `strategy`/`refine` (only the augmented mode exists and augmented QR needs no
refinement); a value update reuses the QR symbolic analysis (`refactor!`) just like the LU
algorithm, and a singular `A` likewise returns `ReturnCode.Infeasible`. It is opt-in — the
default algorithm stays the LU one.

## Public API

```
SparseWithDenseRowColMatrix    SelectorMatrix
sparsepart  fillpart  lowrankfactors  exclusive_sparsepart  denserank
factorize   lu   qr   \\   ldiv!   refactor!   lu!   qr!   update_lowrank!
lstsq   SparseWithDenseRowColLeastSquares   # minimum-norm least squares (one-shot; cached/reusable)
SparseWithDenseRowColWoodbury  SparseWithDenseRowColAugmented  SparseWithDenseRowColQRAugmented
recommend_lowrank_peel   PeelRecommendation
SparseWithDenseRowColFactorization     # LinearSolve.jl algorithm (LU/Woodbury, default)
SparseWithDenseRowColQRFactorization   # LinearSolve.jl algorithm (stable QR, opt-in)
```

The `:iterative` engine of `lstsq` requires `using IterativeSolvers` (a package extension).

## Installation

PureKLU.jl is not yet in the General registry; install it from the SciML repository:

```julia
using Pkg
Pkg.add(url="https://github.com/SciML/PureKLU.jl")
Pkg.add(url="https://github.com/SciML/SparseWithDenseRowColMatrices.jl")
```
