module RNAWorld

using OrdinaryDiffEqRosenbrock
using OrdinaryDiffEqVerner
using OptimizationBBO
using DataInterpolations
using Symbolics
using Plots
using Optim
using ForwardDiff
using LinearAlgebra

"""
Model specification.
"""
struct ReactionModel
    # differential equations that describe the model
    state_dynamics!
    adjoint_dynamics!
    ks
end

function Hamiltonian(model::ReactionModel, n, λ, V)
    (; state_dynamics!, ks) = model
    dxdt = zeros(length(n))
    state_dynamics!(dxdt, n, ks, V)
    λ ⋅ dxdt
end

# state_dynamics! -> Hamiltonian
# forward_backward_sweep(rate_constants, state_dynamics!) -> optimal V


include("plot_utils.jl")
include("forward_backward_sweep.jl")

export volume_simulation, ReactionModel, Hamiltonian
export simulation_plot, volume_plot, all_species_plot, RNA_plot
end
