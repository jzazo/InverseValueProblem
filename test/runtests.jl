include(joinpath(@__DIR__, "..", "src", "ADMM.jl"))
include(joinpath(@__DIR__, "..", "src", "ClassificationInverseValueProblem.jl"))
using .ADMM
using .ClassificatinoInerseValueProblem

using LinearAlgebra
using JuMP, OSQP
using Test

function softthreshold(x::T, λ::T)::T where {T <: AbstractFloat}
    @assert λ >= 0 "Argument λ must exceed 0"
    x > λ && return (x - λ)
    x < -λ && return (x + λ)
    return zero(T)
end

"Solve LASSO problem for small matrix A ∈ R^{N×d} and sparse vector x."
function test_admm()
    N, d = 4, 5
    A = randn(N, d)
    x = zeros(d)
    x[1] = 1.0
    y = A * x
    λ = 0.001  # Lasso regularization weight
    ρ = 1.0  # ADMM proximal weight
    ϵ = 1e-5  # ADMM tolerance

    prox_f = x -> softthreshold.(x, λ / ρ)
    prox_g = x -> (A'A + ρ * I) \ (A'y + ρ * x)

    xr = ADMM.admm(zero(x), zero(x), prox_f, prox_g, ϵ)
    @test x ≈ xr atol = 1e-2
end

test_admm()


"""Minimizes f(δ, π) = δ + 1/(2λ) ‖ π - πk ‖^2, s.t., π'b - z ≥ δ, δ ≥ 0."""
function test_min_fpidelta()
    n = 20
    πk = randn(n)
    bi = randn(n)
    zi = rand()
    λ = 1.0

    model = Model(OSQP.Optimizer)
    @variable(model, δi >= 0)
    @variable(model, πi[1:n])
    @objective(model, Min, δi + 1 / (2 * λ) * sum((πi - πk).^2))
    @constraint(model, con, πi' * bi - zi - δi >= 0.0)
    optimize!(model)

    πi_sol1 = value.(πi)
    πi_sol2 = ClassificatinoInerseValueProblem.min_fpidelta(πk, bi, zi, λ)
    @test πi_sol1 ≈ πi_sol2 atol = 1e-4 * n
end

test_min_fpidelta()