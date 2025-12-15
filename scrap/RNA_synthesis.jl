using Pkg
Pkg.add(["Catalyst", "DifferentialEquations", "Plots"])

using Catalyst
#using DifferentialEquations
using OrdinaryDiffEqDefault
using Plots

# Define the species
#=
Define the species:
Cp is the inactivated species
ACp is the activated species
RNA is the RNA strand  
A is 2-aminoimidazole
CppC is the catalyzed dimer (1,3-di(Cystine-5′-phosphoro)-2-aminoimidazolium)
=#
@variables t
@species Cp(t) RNA(t) CppC(t) ACp(t) A(t)
@parameters V(t)
@paramters K_m v_max 

# Define the reactions
rxns = @reaction_network begin
    k1/V^2, 2ACp → CppC + A      # Formation of the catalyst (dimer of Cp)
    mm(CppC, v_max, K_m), CppC → Cp + RNA      # Catalyzed polymer growth
    k3, Cp + A → ACp         # Formation of Activated Cystine
    k4, CppC → ACp + Cp      # Degradation of the Catalyzed dimer
    k5, ACp → Cp + A         # Degradation of the Activated Cystine
    k6, CppC + A → 2ACp      # Degradation of the catalyzed dimer with 2-aminoimidazole
end

# Initial conditions for the species
u0 = [ACp => 100.0, CppC => 0.0, A => 0.0, RNA => 1.0, Cp => 0.0]

# Parameters for the reactions (rate constants)
p = [1.0, 0.1, 0.05, 0.02, 0.03, 0.01]  # [k1, k2, k3, k4, k5, k6]

# Define the time span for the simulation
tspan = (0.0, 50.0)

# Desired polymer size in number of molecules
desired_polymer_size = 50  # This is the target size (number of RNA molecules)
monomer_size = 1.0  # The size of each monomer unit

# Define a callback to increase and then decrease concentrations
function condition_increase(u, t, integrator)
    mod(t, 20.0) == 10.0  # Increase every 20 time units, for the first 10 units
end

function affect_increase!(integrator)
    integrator.u[1:4] *= 10.0  # Increase all concentrations tenfold
end

function condition_decrease(u, t, integrator)
    mod(t, 20.0) == 0.0 # Decrease every 20 time units, at the 10th unit
end

function affect_decrease!(integrator)
    integrator.u[1:4] /= 10.0  # Decrease all concentrations back to original
end

cb_increase = DiscreteCallback(condition_increase, affect_increase!)
cb_decrease = DiscreteCallback(condition_decrease, affect_decrease!)

# Create the ODE problem
prob = ODEProblem(rxns, u0, tspan, p)

# Solve the ODE problem with the callbacks
sol = solve(prob, Tsit5(), callback=CallbackSet(cb_increase, cb_decrease))

# Calculate the absolute number of RNA molecules over time
rna_count = sol[RNA]

# Find the time it takes to reach the desired RNA size
time_to_reach_size = findfirst(>(desired_polymer_size), rna_count)

# Plot the results
plot(sol, vars=[Cp, RNA, CppC, ACp, A], xlabel="Time", ylabel="Concentration", legend=true)
title!("Polymer Growth with Catalyst and Wet-Dry Cycles")

# Plot the size of the polymer over time
plot(sol.t, rna_count, xlabel="Time", ylabel="Number of RNA Molecules", legend=false)
title!("Number of RNA Molecules Over Time")

# Display the time to reach the desired RNA size
println("Time to reach the desired polymer size ($desired_polymer_size units): $time_to_reach_size")
