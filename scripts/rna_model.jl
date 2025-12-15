include("../RNAWorld.jl")
using .RNAWorld
using Symbolics

# Constants
@variables t
@variables α(t) Cp(t) αCp(t) CppC(t) RNA(t)
@variables V(t)
@variables λ₁ λ₂ λ₃ λ₄ λ₅
@variables begin
    k1 = 4.55e-3
    k2 = 0.183
    k3 = 0.167
    k4 = 0.1
    k5 = 3.33e-3
    Kd = 24.7
    c6 = 1.06
end

const ∂ = Symbolics.derivative
const r1 = k1*αCp^2 / V^2
const r2 = k2*α*CppC / V^2
const r3 = k3*CppC / V
const r4 = k4*α*Cp / V^2
const r5 = k5*αCp / V
const r6 = c6 * (1 + (Cp + αCp) / (Kd * V)) * CppC / V
#const r7 = k6*RNA

const H = V * (
        λ₁ * (r1 + r5 - r2 - r4) +
        λ₂ * (r3 + r5 - r4) +
        λ₃ * (2r2 - 2r1 + r3 + r4 - r5) +
        λ₄ * (r1 - r2 - r3 - r6) + 
        λ₅ * r6
    )
const λₜ = -[∂(H, α), ∂(H, Cp), ∂(H, αCp), ∂(H, CppC), ∂(H, RNA)]
const nₜ = [∂(H, λ₁), ∂(H, λ₂), ∂(H, λ₃), ∂(H, λ₄), ∂(H, λ₅)]

λ = [λ₁, λ₂, λ₃, λ₄, λ₅]
n = [α, Cp, αCp, CppC, RNA]
k = [k1, k2, k3, k4, k5, Kd, c6]

adjoint_dynamics, adjoint_dynamics! = Symbolics.build_function(λₜ, n, λ, k, V; expression = Val{false})
state_dynamics, state_dynamics! = Symbolics.build_function(nₜ, n, k, V; expression = Val{false})

#############################
###### SIMULATION ###########
#############################
const V_lo = 0.1
const V_hi = 10.

begin
    n0 = [10.0, 10.0, 10.0, 10.0, 0.0]    # initial state
    λ0 = [0.0, 0.0, 0.0, 0.0, 1.0]   # final costate
    ks = (; k1 = 4.55e-3, k2 = 0.183, k3 = 0.167, k4 = 0.1, k5 = 3.33e-3, Kd = 24.7, c6 = 1.06)
    tspan = (0.0, 20.0)
    
    rnamodel = RNAWorld.ReactionModel(state_dynamics!, adjoint_dynamics!, ks)
    V_opt, sols = RNAWorld.volume_simulation(rnamodel, V_lo, V_hi, n0, λ0, tspan)
    plt = RNAWorld.simulation_plot(sols, V_opt; Vmin = V_lo, Vmax = V_hi)
    savefig(plt, "figs/rna_model.png")
end


####################################
###### Parameter optimization ######
####################################

ode_func_low!(du, u, p, t) = state_dynamics!(du, u, ks, V_lo)
ode_func_hi!(du, u, p, t) = state_dynamics!(du, u, ks, V_hi)

# Highest at any point?
function run_trial(ks, ps)
    ks = exp.(ks)
    x0, λ0 = ps[1, :], ps[2, :]
    controller, sol_optim, sol_co = ForwardBackwardSweep(init_controller, Vern7(), x0, λ0, ks, tspan)
    prob_low = ODEProblem(ode_func_low!, x0, tspan)
    sol_lo = solve(prob_low, Vern7())

    return sol_lo.u[end][end] - sol_optim.u[end][end]
end

prob = OptimizationProblem(run_trial, log.(ks), p, lb = fill(-7, 5), ub = fill(7, 5))
sol = solve(prob, BBO_xnes())
