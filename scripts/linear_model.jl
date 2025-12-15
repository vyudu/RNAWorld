include("../RNAWorld.jl")
using .RNAWorld
using Symbolics

const Vmin, Vmax = 0.1, 10.0

@variables t
@variables A(t) B(t) C(t)
@variables V(t)
@variables λ[1:3]
@variables k1 k2 k3 Ki Kd D
const ∂ = Symbolics.derivative
const r1 = V * k1 * (A/V) / (1 + (C/V) / Ki)
const r2 = V * k2 * (B/V) / (1 + (D/V) / Kd)
const r3 = V * k3 * (C/V)

const H = -λ[1] * r1 + λ[2] * (r1 - r2) + λ[3] * (r2 - r3)
const λₜ = -[∂(H, A), ∂(H, B), ∂(H, C)]
const nₜ = [∂(H, λ[1]), ∂(H, λ[2]), ∂(H, λ[3])]

k = [k1, k2, k3, Ki, Kd, D]
n = [A, B, C]

adjoint_dynamics, adjoint_dynamics! = Symbolics.build_function(λₜ, n, λ, k, V; expression = Val{false})
state_dynamics, state_dynamics! = Symbolics.build_function(nₜ, n, k, V; expression = Val{false})

# Simulation
begin
    # Initial conditions
    n0 = [200.0, 0.0, 0.0]  # [A, B, C]
    λT = [0.0, 0.0, 1.0]   # final adjoint state: sensitivity to C

    ks = (; k1 = 1.9, k2 = 0.4, k3 = 1.7, Ki = 0.9, Kd = 0.6, D = 8.7)
    tspan = (0.0, 4.0)

    linear = ReactionModel(state_dynamics!, adjoint_dynamics!, ks)
    V_opt, sols = volume_simulation(linear, Vmin, Vmax, n0, λT, tspan)

    plt = simulation_plot(sols, V_opt; idx = 3, Vmin, Vmax)
    savefig(plt, "figs/linear_model.png")
end
