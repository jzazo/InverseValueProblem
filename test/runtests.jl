include(joinpath(@__DIR__, "..", "src", "ADMM.jl"))
using .ADMM

using LinearAlgebra
using Test

function softthreshold(x::T, λ::T)::T where {T <: AbstractFloat}
    @assert λ >= 0 "Argument λ must exceed 0"
    x > λ && return (x - λ)
    x < -λ && return (x + λ)
    return zero(T)
end

"Project x=6 onto x^2-4x-4<=0; solution is ."
function test_admm()
    N, d = 4, 5
    A = randn(N, d)
    x = zeros(d)
    x[1] = 1.0
    y = A * x
    λ = 0.001  # Lasso regularization weight
    ρ = 1.0  # ADMM proximal weight
    ϵ = 1e-5

    prox_f = x -> softthreshold.(x, λ / ρ)
    prox_g = x -> (A'A + ρ * I) \ (A'y + ρ * x)

    xr = ADMM.admm(zero(x), zero(x), prox_f, prox_g, ϵ)
    @test x ≈ xr atol = 1e-2
end

test_admm()