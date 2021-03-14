module ClassificatinoInerseValueProblem

using Einsum
using LIBLINEAR: linear_train

"""Minimizes f(δi, πi) = δi + 1/(2λ) ‖ πi - πk ‖^2, s.t., πi'bi - zi ≥ δi, δi ≥ 0."""
function min_fpidelta(
    πk::AbstractVector{T}, bi::AbstractVector{T}, zi::T, λ::T
) where {T <: AbstractFloat}
    res = max(zi - bi'πk, 0.0) / (bi'bi) / λ
    μi = min(res, 1)
    πi = πk + λ * μi * bi
    return πi
end


"""Minimizes f(c, ϵ) = ϵ'1 + α ‖ c - cα ‖^2, s.t., ϵ ≥ 0, c ∈ C, xi'c ≤ zi ∀ i."""
function min_cepsilon(
    xi::Matrix{T}, zi::Vector{T}, ck::Vector{T}, cα::Vector{T}, λ::T, α::T
) where {T <: AbstractFloat}
    ckα = 1 / (1 + λ * α) * (ck + λ * α * cα)
    cost = λ / (1 + λ * α)
    @einsum b[i] := xi[j, i] * ckα[j] - zi[i]
    W = b[b .> 0.0]

    if length(W) == 0
        return zero(ck)
    else
        a = -1 / W * xi[:, b .> 0.0]
        labels = ones(length(W))
        model = linear_train(labels, a; eps=1e-4, solver_type=Cint(3), C=cost, bias=-1, W=W)
        return model.w + ckα
    end
end


end # module