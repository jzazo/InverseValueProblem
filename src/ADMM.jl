module ADMM

using LinearAlgebra: norm

"""
    admm(x0, z0, u0, prox_f, prox_g, ϵ=1e-4)

Alternating Direction Method of Multipliers for the sum of two functions f and g
(also known as Douglas-Rachford algorithm).

# Arguments
- `x0::Vector`: starting estimate for x primal variable.
- `z0::Vector`: starting estimate for z consensus variable.
- `u0::Vector`: starting estimate for u dual variable.

"""
function admm(
    x0::Vector{T}, u0::Vector{T}, prox_f, prox_g, ϵ::T=1e-4
) where {T <: AbstractFloat}
    xk, uk = copy(x0), copy(u0)
    xkk, zkk, ukk = xk .+ ϵ, xk .+ ϵ, uk .+ ϵ

    while norm(xkk - xk) / norm(xkk) + norm(ukk - uk) / norm(ukk) >= ϵ
        xk, uk = copy(xkk), copy(ukk)

        zkk = prox_g(xk + uk)
        xkk = prox_f(zkk - uk)
        ukk = uk + xkk - zkk
    end
    return xkk
end

end