# ------------------
# Woodbury factorization (primary path)
# ------------------
#
#   A = S + U V
#   A⁻¹ b = S⁻¹b − Z C⁻¹ (V S⁻¹b),   Z = S⁻¹U (n×r),   C = I_r + V Z (r×r)
#
# Factor S once with PureKLU; precompute Z (one multi-RHS solve) and lu(C). Each subsequent
# solve costs one sparse solve + an r×r dense solve. `refactor!` reuses S's symbolic
# analysis (PureKLU `klu!`) so a Newton sweep (same pattern, new values) never re-analyzes.

"""
    SparseWithDenseRowColWoodbury{T} <: LinearAlgebra.Factorization{T}

Sherman–Morrison–Woodbury factorization of an [`SparseWithDenseRowColMatrix`](@ref) `A = S + U V`,
built from a single [`PureKLU`](https://github.com/SciML/PureKLU.jl) LU factorization of the
sparse part `S` plus a dense `r × r` correction `C = I + V S⁻¹U`. Created by
[`factorize`](@ref)/[`lu`](@ref) (the default strategy when `S` is nonsingular). Solve with
`\\`/`ldiv!`; update in place for a new set of numeric values with [`refactor!`](@ref).

Valid only when both `S` and `C` are nonsingular; [`factorize`](@ref) falls back to
[`SparseWithDenseRowColAugmented`](@ref) otherwise. The forward error of the Woodbury solve scales
with `κ(S)·κ(C)` rather than `κ(A)`, so one step of iterative refinement (`refine` keyword,
default `1`) is applied to restore backward stability — this is allocation-free, reusing the
cached factors.
"""
mutable struct SparseWithDenseRowColWoodbury{T, KF, TS, TU, TV, CF} <: LinearAlgebra.Factorization{T}
    Sfact::KF            # PureKLU.KLUFactorization of S (symbolic reused across refactors)
    Sown::TS             # owned SparseMatrixCSC copy of S — values for the residual matvec
    U::TU                # n×r
    V::TV                # r×n
    Z::Matrix{T}         # n×r = S⁻¹U, doubles as the multi-RHS solve! buffer (unit stride)
    C::Matrix{T}         # r×r scratch holding (after lu!) the LU factors of I + V*Z
    Cfact::CF            # lu!(C)
    ywork::Vector{T}     # length n — holds S⁻¹b
    tr::Vector{T}        # length r — V*y and C⁻¹(·)
    res::Vector{T}       # length n — structured residual / correction
    borig::Vector{T}     # length n — saved RHS for iterative refinement
    r::Int
    refine::Int
    illconditioned::Bool
end

function SparseWithDenseRowColWoodbury(
        Sfact, Sown, U, V, Z::Matrix{T}, C, Cfact, ywork, tr, res, borig, r, refine, ill
    ) where {T}
    return SparseWithDenseRowColWoodbury{
        T, typeof(Sfact), typeof(Sown), typeof(U), typeof(V), typeof(Cfact),
    }(Sfact, Sown, U, V, Z, C, Cfact, ywork, tr, res, borig, r, refine, ill)
end

Base.size(F::SparseWithDenseRowColWoodbury) = (size(F.Sown, 1), size(F.Sown, 2))
Base.size(F::SparseWithDenseRowColWoodbury, i::Integer) = size(F.Sown, i)
LinearAlgebra.issuccess(F::SparseWithDenseRowColWoodbury) =
    LinearAlgebra.issuccess(F.Sfact) && LinearAlgebra.issuccess(F.Cfact)
denserank(F::SparseWithDenseRowColWoodbury) = F.r

# Underlying-float type used to pick conditioning / refinement tolerances.
_tol_float(::Type{T}) where {T <: AbstractFloat} = T
_tol_float(::Type{Complex{T}}) where {T <: AbstractFloat} = T
_tol_float(::Type{T}) where {T} = Float64    # ForwardDiff.Dual, Rational, …

# Build a freshly-owned SparseMatrixCSC{T,Ti} (Ti a KLU index type) we are free to mutate.
# `Int` is always a valid KLU index type on a given build (Int32 on 32-bit, Int64 on 64-bit),
# so it is the safe fallback when S's index type is not accepted by PureKLU on this platform.
function _own_sparse(::Type{T}, S::SparseArrays.AbstractSparseMatrixCSC) where {T}
    Si = SparseMatrixCSC(S)
    Ti = eltype(SparseArrays.getcolptr(Si))
    if Ti <: PureKLU.KLUITypes
        return eltype(Si) === T ? copy(Si) : SparseMatrixCSC{T, Ti}(Si)
    else
        return SparseMatrixCSC{T, Int}(Si)
    end
end

# Z ← S⁻¹U  (reuses Sfact's current numeric factorization).
function _recompute_Z!(F::SparseWithDenseRowColWoodbury)
    materialize_U!(F.Z, F.U)
    PureKLU.solve!(F.Sfact, F.Z)
    return F
end
# C ← lu(I + V*Z)  (assumes F.Z is current).
function _recompute_C!(F::SparseWithDenseRowColWoodbury{T}) where {T}
    if F.r > 0
        mul!(F.C, F.V, F.Z)
        @inbounds for i in 1:F.r
            F.C[i, i] += one(T)
        end
    end
    F.Cfact = lu!(F.C; check = false)
    LinearAlgebra.issuccess(F.Cfact) || throw(SingularException(0))
    return F
end
# Z then C. Assumes Sfact has just been (re)factored.
_recompute_correction!(F::SparseWithDenseRowColWoodbury) = _recompute_C!(_recompute_Z!(F))

function _woodbury(A::SparseWithDenseRowColMatrix{T}; refine::Integer = 1) where {T}
    n = size(A, 1)
    r = denserank(A)
    Sown = _own_sparse(T, A.S)
    # PureKLU never throws on a singular factor: it stops at the zero pivot and reports
    # `issuccess == false` (factoring further, or solving, reads past the truncated factor and
    # hits a BoundsError). Surface it as a SingularException so `factorize`'s `:auto` path can
    # fall back to the augmented system.
    Sfact = PureKLU.klu(Sown)
    LinearAlgebra.issuccess(Sfact) || throw(SingularException(0))
    # Own copies of the low-rank factors (S is already copied via `_own_sparse`); this keeps
    # the factorization independent of later in-place mutation of the input's U/V, matching
    # the augmented path. SelectorMatrix is immutable, so `_copy_U` just rebuilds it.
    U, V = _copy_U(A.U), copy(A.V)
    Z = Matrix{T}(undef, n, r)
    materialize_U!(Z, U)
    PureKLU.solve!(Sfact, Z)               # Z = S⁻¹U  (one multi-RHS solve)
    C = Matrix{T}(undef, r, r)
    nrmC = zero(real(T))
    if r > 0
        mul!(C, V, Z)
        @inbounds for i in 1:r
            C[i, i] += one(T)
        end
        nrmC = opnorm(C, 1)
    end
    Cfact = lu!(C; check = false)
    LinearAlgebra.issuccess(Cfact) || throw(SingularException(0))
    ill = r > 0 && nrmC * opnorm(inv(Cfact), 1) > inv(sqrt(eps(_tol_float(T))))
    return SparseWithDenseRowColWoodbury(
        Sfact, Sown, U, V, Z, C, Cfact,
        Vector{T}(undef, n), Vector{T}(undef, r), Vector{T}(undef, n), Vector{T}(undef, n),
        r, Int(refine), ill
    )
end

# ------------------
# Solve
# ------------------

# b := A⁻¹ b via Woodbury (no refinement). Allocation-free.
@views function _woodbury_solve!(F::SparseWithDenseRowColWoodbury{T}, b::AbstractVector) where {T}
    copyto!(F.ywork, b)
    PureKLU.solve!(F.Sfact, F.ywork)          # ywork = S⁻¹b
    if F.r > 0
        mul!(F.tr, F.V, F.ywork)              # tr = V S⁻¹b
        ldiv!(F.Cfact, F.tr)                  # tr = C⁻¹ V S⁻¹b
        copyto!(b, F.ywork)
        mul!(b, F.Z, F.tr, -one(T), one(T))   # b = S⁻¹b - Z C⁻¹ V S⁻¹b
    else
        copyto!(b, F.ywork)
    end
    return b
end

# res := A*x using the owned sparse part and the low-rank factors. Allocation-free.
function _applyA!(F::SparseWithDenseRowColWoodbury{T}, res::AbstractVector, x::AbstractVector) where {T}
    mul!(res, F.Sown, x)
    if F.r > 0
        mul!(F.tr, F.V, x)
        mul!(res, F.U, F.tr, one(T), one(T))
    end
    return res
end

function LinearAlgebra.ldiv!(F::SparseWithDenseRowColWoodbury{T}, b::AbstractVector) where {T}
    length(b) == size(F, 1) || throw(DimensionMismatch())
    F.refine > 0 && copyto!(F.borig, b)
    _woodbury_solve!(F, b)
    @inbounds for _ in 1:F.refine
        _applyA!(F, F.res, b)                 # res = A x
        @. F.res = F.borig - F.res            # res = b - A x   (structured residual)
        _woodbury_solve!(F, F.res)            # res = A⁻¹ residual
        @. b = b + F.res                      # x += correction
    end
    return b
end

# Matrix RHS: solve column by column (reuses the alloc-free vector path + refinement).
function LinearAlgebra.ldiv!(F::SparseWithDenseRowColWoodbury, B::AbstractMatrix)
    size(B, 1) == size(F, 1) || throw(DimensionMismatch())
    for c in axes(B, 2)
        ldiv!(F, view(B, :, c))
    end
    return B
end

# Out-of-place forms (split vector/matrix so each is strictly more specific than the generic
# `ldiv!(::AbstractVecOrMat, ::Factorization, ::AbstractVecOrMat)`). `\` is provided by
# LinearAlgebra's generic `Factorization` fallback, which routes here via `ldiv!` and also
# supplies the complex-RHS-on-real-factorization split.
LinearAlgebra.ldiv!(y::AbstractVector, F::SparseWithDenseRowColWoodbury, b::AbstractVector) =
    ldiv!(F, copyto!(y, b))
LinearAlgebra.ldiv!(Y::AbstractMatrix, F::SparseWithDenseRowColWoodbury, B::AbstractMatrix) =
    ldiv!(F, copyto!(Y, B))

# ------------------
# Adjoint / transpose solve  (Aᴴ = Sᴴ + Vᴴ Uᴴ):  A⁻ᴴ b = S⁻ᴴ(b − Vᴴ C⁻ᴴ Zᴴ b)
# ------------------
# `F'` / `transpose(F)` on a Factorization wrap it in Adjoint/TransposeFactorization (newer
# Julia) or Adjoint/Transpose (older). `\` for those wrappers is provided generically by
# LinearAlgebra and routes to the `ldiv!` methods below.

const AdjointFact =
    isdefined(LinearAlgebra, :AdjointFactorization) ? LinearAlgebra.AdjointFactorization : LinearAlgebra.Adjoint
const TransposeFact =
    isdefined(LinearAlgebra, :TransposeFactorization) ? LinearAlgebra.TransposeFactorization : LinearAlgebra.Transpose

for (Wrap, op, klu_adj) in (
        (AdjointFact, :adjoint, :(F.Sfact')),
        (TransposeFact, :transpose, :(transpose(F.Sfact))),
    )
    @eval function LinearAlgebra.ldiv!(
            Fw::$Wrap{<:Any, <:SparseWithDenseRowColWoodbury{T}}, b::AbstractVector
        ) where {T}
        F = parent(Fw)
        length(b) == size(F, 1) || throw(DimensionMismatch())
        if F.r > 0
            mul!(F.tr, $op(F.Z), b)            # tr = Zᴴ b
            ldiv!($op(F.Cfact), F.tr)          # tr = C⁻ᴴ tr
            mul!(F.res, $op(F.V), F.tr)        # res = Vᴴ tr
            @. F.res = b - F.res
        else
            copyto!(F.res, b)
        end
        PureKLU.solve!($klu_adj, F.res)        # res = S⁻ᴴ(…)
        copyto!(b, F.res)
        return b
    end
    @eval function LinearAlgebra.ldiv!(
            Fw::$Wrap{<:Any, <:SparseWithDenseRowColWoodbury}, B::AbstractMatrix
        )
        for c in axes(B, 2)
            ldiv!(Fw, view(B, :, c))
        end
        return B
    end
end

# ------------------
# Complex RHS over a real factorization: solve real and imaginary parts separately, since
# the cached scratch buffers are real-typed. Mirrors LinearAlgebra's dense LU and PureKLU.
# ------------------

function _ldiv_realfact_complex!(Ft, b::AbstractVector{<:Complex})
    br = real.(b)
    bi = imag.(b)
    ldiv!(Ft, br)
    ldiv!(Ft, bi)
    @inbounds @. b = complex(br, bi)
    return b
end

LinearAlgebra.ldiv!(F::SparseWithDenseRowColWoodbury{<:Real}, b::AbstractVector{<:Complex}) =
    _ldiv_realfact_complex!(F, b)
LinearAlgebra.ldiv!(Fw::AdjointFact{<:Any, <:SparseWithDenseRowColWoodbury{<:Real}}, b::AbstractVector{<:Complex}) =
    _ldiv_realfact_complex!(Fw, b)
LinearAlgebra.ldiv!(Fw::TransposeFact{<:Any, <:SparseWithDenseRowColWoodbury{<:Real}}, b::AbstractVector{<:Complex}) =
    _ldiv_realfact_complex!(Fw, b)

# ------------------
# Refactor (Newton hot path): reuse symbolic analysis, new numeric values
# ------------------

"""
    refactor!(F::SparseWithDenseRowColWoodbury, A::SparseWithDenseRowColMatrix; check=true) -> F
    refactor!(F::SparseWithDenseRowColWoodbury, S::AbstractSparseMatrixCSC; fill=nothing, check=true) -> F
    refactor!(F::SparseWithDenseRowColWoodbury, Snzval::AbstractVector; fill=nothing) -> F

Update the factorization `F` in place with new numeric values that share the **same
sparsity pattern** as the matrix `F` was built from, reusing PureKLU's symbolic analysis and
preallocated workspace (no re-analysis). This is the hot path for Newton iterations on a
boundary-value problem, where the Jacobian's pattern is fixed and only its values change.

The cheapest form takes the raw `nzval` vector of the new sparse part (no pattern check —
the caller guarantees the pattern is unchanged); pass `fill` to also update the low-rank
right factor `V`. The `SparseMatrixCSC`/`SparseWithDenseRowColMatrix` forms validate the pattern when
`check=true`.
"""
function refactor!(F::SparseWithDenseRowColWoodbury, Snzval::AbstractVector; fill = nothing)
    nz = SparseArrays.nonzeros(F.Sown)
    length(Snzval) == length(nz) ||
        throw(DimensionMismatch("new nzval has length $(length(Snzval)); pattern has $(length(nz))"))
    copyto!(nz, Snzval)
    PureKLU.klu!(F.Sfact, nz)                  # reuse symbolic + workspace
    fill !== nothing && copyto!(F.V, fill)
    _recompute_correction!(F)
    return F
end

function refactor!(
        F::SparseWithDenseRowColWoodbury, S::SparseArrays.AbstractSparseMatrixCSC;
        fill = nothing, check::Bool = true
    )
    if check
        _same_pattern(F.Sown, S) ||
            throw(ArgumentError("refactor! requires the new sparse part to have the same sparsity pattern."))
    end
    return refactor!(F, SparseArrays.nonzeros(S); fill)
end

function refactor!(F::SparseWithDenseRowColWoodbury, A::SparseWithDenseRowColMatrix; check::Bool = true)
    size(A.V) == size(F.V) ||
        throw(DimensionMismatch("the fill rank/shape changed; rebuild with `factorize`."))
    if F.U isa SelectorMatrix
        # A selector U has no stored values to update; refuse to silently ignore a switch to
        # a different (e.g. dense) left factor, which would feed a stale U to the solve.
        (A.U isa SelectorMatrix && size(A.U) == size(F.U)) ||
            throw(ArgumentError("refactor! cannot change the left factor U from a SelectorMatrix to a dense/incompatible one; rebuild with `factorize`."))
    else
        copyto!(F.U, A.U)
    end
    return refactor!(F, A.S; fill = A.V, check)
end

function _same_pattern(A::SparseArrays.AbstractSparseMatrixCSC, B::SparseArrays.AbstractSparseMatrixCSC)
    return size(A) == size(B) &&
        SparseArrays.getcolptr(A) == SparseArrays.getcolptr(B) &&
        SparseArrays.rowvals(A) == SparseArrays.rowvals(B)
end

"""
    update_lowrank!(F::SparseWithDenseRowColWoodbury; U=nothing, V=nothing) -> F

Update only the low-rank factors of `F`, **reusing the cached factorization of the sparse
part `S`** (no re-factorization of `S`). This is the Sherman–Morrison–Woodbury fast path for
a fixed sparse base solved repeatedly under a *changing low-rank correction* — varying
boundary conditions / coupling, adding or removing a constraint, sensitivity or parameter
sweeps. Factor `S` once with [`factorize`](@ref), then per step `update_lowrank!(F; V=…)`
and solve.

Passing `V` (the `r × n` dense factor) recomputes only the `r × r` correction `C = I + V Z`.
Passing `U` additionally recomputes `Z = S⁻¹U` (one multi-RHS solve against the *cached* `S`
factorization), so omit `U` when the left factor is unchanged. The sparsity pattern and rank
are fixed; to change `S`'s values use [`refactor!`](@ref), and to change the rank rebuild
with [`factorize`](@ref).
"""
function update_lowrank!(F::SparseWithDenseRowColWoodbury; U = nothing, V = nothing)
    if U !== nothing
        if F.U isa SelectorMatrix
            (U isa SelectorMatrix && size(U) == size(F.U)) ||
                throw(ArgumentError("the left factor U is a SelectorMatrix; pass a matching SelectorMatrix or rebuild with `factorize`."))
        else
            copyto!(F.U, U)
        end
        _recompute_Z!(F)                      # reuse S's cached factorization — no klu!(S)
    end
    V !== nothing && copyto!(F.V, V)
    return _recompute_C!(F)
end
