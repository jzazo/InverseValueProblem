module ADMM

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
    x0::Vector,
    z0::Vector,
    u0::Vector,
    prox_f::Function,
    prox_g::Function,
    ϵ::AbstractFloat=1e-4,
)
    xk, zk, uk = copy(x0), copy(z0), copy(u0)
    xkk, zkk, ukk = xk + ϵ, zk + ϵ, uk + ϵ

    while (
        norm(xkk - xk) / norm(xkk) +
        norm(zkk - zk) / norm(zkk) +
        norm(ukk - uk) / norm(ukk) >= ϵ
    )
        xk, zk, uk .= copy(xkk), copy(zkk), copy(ukk)

        xkk = prox_f(zk - uk)
        zkk = prox_g(xkk + uk)
        ukk = uk + xkk - zkk
    end
    return xkk
end

end