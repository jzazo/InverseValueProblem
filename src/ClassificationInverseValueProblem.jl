module ClassificatinoInerseValueProblem

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


"""Minimizes f(ϵ, c; ck, cα, λ, xi) = ϵ'1 + 1/(2λ) ‖c - ck‖^2 + α/2 ‖c - cα‖^2,
s.t., c'xi - zi ≤ ϵi, ϵi ≥ 0, for all i, c ∈ C.

"""
function min_fcepsilon(
    xi::Matrix{T}, zi::Vector{T}, ck::Vector{T}, cα::Vector{T}, λ::T, α::T
) where {T <: AbstractFloat}
    ckα = 1 / (1 + λ * α) * (ck + λ * α * cα)
    cost = λ / (1 + λ * α)

    # Create new array bi = xi'ckα - zi for all i
    b = zero(zi)
    for j in 1:size(xi, 1), i in 1:size(xi, 2)
        b[i] += xi[j, i] * ckα[j]
    end
    b .-= zi
    W = b[b .> 0.0]

    if length(W) == 0
        return zero(ck)
    else
        j = 0
        a = Matrix{Float64}(undef, size(xi, 1), length(W))
        for i in 1:length(b)
            if b[i] > 0.0
                j += 1
                a[:, i] = -1 / W[j] * xi[:, i]
            end
        end
        labels = ones(length(W))
        model = linear_train(labels, a; eps=1e-4, solver_type=Cint(3), C=cost, bias=-1, W=W)
        return model.w + ckα
    end
end


end # module