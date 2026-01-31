using Pkg
Pkg.activate(".")
using DifferentialEquations, Optim, Random, Plots, Statistics

# Parameters
k1, k2, k3 = 1.6, 1 , 1
Ki = 2
Kd = 0.3
D = 6.7
Vmin, Vmax = 1.0, 100.0
ϵ = 1e-8  # small value to avoid division by zero

# Rates with stabilizers
# Restore C inhibition to A->B rate
rate1(A, C, V) = V * k1 * (A / (V + ϵ)) / (1 + (C / (V + ϵ)) / Ki)
rate2(B, V, D) = V * k2 * (B / (V + ϵ)) / (1 + (D / (V + ϵ)) / Kd)
rate3(C, V) = V * k3 * (C / (V + ϵ))

# System dynamics: n = [A, B, C]
function state_dynamics!(du, n, p, t, λ, V, D)
    A, B, C = n
    r1 = rate1(A, C, V)
    r2 = rate2(B, V, D)
    r3 = rate3(C, V)
    du[1] = -r1
    du[2] = r1 - r2
    du[3] = r2 - r3
end

# Hamiltonian
function H(n, λ, V, D)
    A, B, C = n
    L1, L2, L3 = λ
    r1 = rate1(A, C, V)
    r2 = rate2(B, V, D)
    r3 = rate3(C, V)
    return L1 * (-r1) + L2 * (r1 - r2) + L3 * (r2 - r3)
end

# Adjoint dynamics
function adjoint_dynamics!(dl, λ, n, V, D)
    dl .= -ForwardDiff.gradient(n -> H(n, λ, V, D), n)
end

# Pontryagin's Maximum Principle: maximize Hamiltonian w.r.t. V
function optimal_V(n, λ)
    obj(V) = -H(n, λ, V, D)  # minimize -H is same as maximize H
    res = optimize(obj, Vmin, Vmax, Brent())
    Vopt = Optim.minimizer(res)
    return clamp(Vopt, Vmin, Vmax)
end

# Forward-backward sweep
function forward_backward_sweep(n0, λT, tspan, tsteps; control_update=optimal_V, iters = 100)
    V_vals = fill((Vmin + Vmax) / 2, length(tsteps))
    sol_n, sol_λ = nothing, nothing
    for sweep in 1:iters
        println("Sweep $sweep")
        λ_dummy = zeros(3)
        # Forward ODE
        function forward_ode!(du, u, p, t)
            idx = findfirst(x -> x ≥ t, tsteps)
            state_dynamics!(du, u, p, t, λ_dummy, V_vals[idx], D)
        end
        prob_fwd = ODEProblem(forward_ode!, n0, tspan)
        sol_n = solve(prob_fwd, Tsit5(), saveat=tsteps)

        # Backward ODE (adjoint)
        function backward_ode!(dl, l, p, t)
            idx = findfirst(x -> x ≥ t, tsteps)
            adjoint_dynamics!(dl, l, sol_n(t), V_vals[idx], D)
        end
        prob_bwd = ODEProblem(backward_ode!, λT, (tspan[2], tspan[1]))
        sol_λ = solve(prob_bwd, Tsit5(), saveat=reverse(tsteps))
        # Update control using optimal V(t)
        for (i, t) in enumerate(tsteps)
            n_t = sol_n(t)
            λ_t = sol_λ(t)
            V_vals[i] = control_update(n_t, λ_t)
        end
    end
    return sol_n, sol_λ, V_vals
end

# Initial conditions
n0 = [200.0, 0.0, 0.0]  # [A, B, C]
λT = [0.0, 0.0, 1.0]   # final adjoint state: sensitivity to C

tspan = (0.0, 4.0)
tsteps = range(tspan[1], tspan[2], length=300)

# Run forward-backward sweep for optimal control
control_Vmax = Returns(Vmax)
control_Vmin = Returns(Vmin)
control_Vmean = Returns((Vmin + Vmax) / 2)

function generate_solutions(n0, λT, tspan, tsteps)
    sol_n, sol_λ, V_vals = forward_backward_sweep(n0, λT, tspan, tsteps);
    sol_n_Vmax, _, _ = forward_backward_sweep(n0, λT, tspan, tsteps; control_update=control_Vmax);
    sol_n_Vmin, _, _ = forward_backward_sweep(n0, λT, tspan, tsteps; control_update=control_Vmin);
    sol_n_Vmean, _, _ = forward_backward_sweep(n0, λT, tspan, tsteps; control_update=control_Vmean);
    
    (; opt = sol_n, max = sol_n_Vmax, min = sol_n_Vmin, mean = sol_n_Vmean)
end

# Plot all C curves together
function plot_concentration_curves(sols; idxs = 3)
    pC = plot(sols.opt, idxs = idxs, label="Optimal V(t)", lw=2)
    plot!(sols.max, idxs = idxs, label="Constant Vmax", lw=2, ls=:dash)
    plot!(sols.min, idxs = idxs, label="Constant Vmin", lw=2, ls=:dot)
    plot!(sols.mean, idxs = idxs, label="Constant Vmean", lw=2, ls=:dashdot)
    xlabel!("Time")
    ylabel!("Concentration C")
    title!("C(t) under different controls")
end

# Plot V(t) for optimal
function plot_volume_curve()
    # This function will be completed in the main block below
    return nothing
end

function plot_vc()
    # This function will be completed in the main block below
    return nothing
# === MAIN BLOCK ===
n0 = [200.0, 0.0, 0.0]
λT = [0.0, 0.0, 1.0]
tspan = (0.0, 4.0)
tsteps = range(tspan[1], tspan[2], length=300)

sols = generate_solutions(n0, λT, tspan, tsteps)

# Extract C(t) for each control
C_opt = [sols.opt(t)[3] for t in tsteps]
C_Vmax = [sols.max(t)[3] for t in tsteps]
C_Vmin = [sols.min(t)[3] for t in tsteps]
C_Vmean = [sols.mean(t)[3] for t in tsteps]

# Extract V(t) for optimal control
_, _, V_vals = forward_backward_sweep(n0, λT, tspan, tsteps)

# Print max C for each control
println("Max C (Optimal): ", maximum(C_opt))
println("Max C (Vmax): ", maximum(C_Vmax))
println("Max C (Vmin): ", maximum(C_Vmin))
println("Max C (Vmean): ", maximum(C_Vmean))

# Plot C curves
plot_concentration_curves(sols)

# Plot V(t) for optimal control
plot(tsteps, V_vals, label="Optimal V(t)", lw=2)
xlabel!("Time")
ylabel!("Volume V(t)")
title!("Optimal Control V(t)")
end

# Plot all together

# Print max C for each control
println("Max C (Optimal): ", maximum(C_opt))
println("Max C (Vmax): ", maximum(C_Vmax))
println("Max C (Vmin): ", maximum(C_Vmin))
println("Max C (Vmean): ", maximum(C_Vmean))



# Detect times when Vmax or Vmin used in numerical optimal control

# If V_num, C_num, C_const are not defined, use V_vals, C_opt, C_Vmean as fallback
# Ensure all required variables are defined in global scope
sols = generate_solutions(n0, λT, tspan, tsteps)
C_opt = [sols.opt(t)[3] for t in tsteps]
C_Vmax = [sols.max(t)[3] for t in tsteps]
C_Vmin = [sols.min(t)[3] for t in tsteps]
C_Vmean = [sols.mean(t)[3] for t in tsteps]
_, _, V_vals = forward_backward_sweep(n0, λT, tspan, tsteps)
V_num = V_vals
C_num = C_opt
C_const = C_Vmean
times_Vmax = [tsteps[i] for i in 1:length(V_num) if isapprox(V_num[i], Vmax; atol=1e-4)]
times_Vmin = [tsteps[i] for i in 1:length(V_num) if isapprox(V_num[i], Vmin; atol=1e-4)]

# Compute and print max concentrations for each control
maxC_num = maximum(C_num)
maxC_const = maximum(C_const)
maxC_Vmax = maximum(C_Vmax)
maxC_Vmin = maximum(C_Vmin)
println("maxC_num = ", maxC_num)
println("maxC_const = ", maxC_const)
println("maxC_Vmax = ", maxC_Vmax)
println("maxC_Vmin = ", maxC_Vmin)

println("Times when Vmax is used in numerical optimal control:")
println(times_Vmax)

println("Times when Vmin is used in numerical optimal control:")
println(times_Vmin)

# Find max C from all controls (numerical optimal and constant)

# Compute maxC_Vmax and maxC_Vmin for summary printout
maxC_num = maximum(C_num)
maxC_const = maximum(C_const)
maxC_Vmax = maximum(C_Vmax)
maxC_Vmin = maximum(C_Vmin)

println("Max concentration C (Numerical Optimal Control): ", maxC_num)
println("Max concentration C (Constant Control): ", maxC_const)

Pkg.add("RecipesBase")
using RecipesBase

function v_control_markers!(p, times_Vmax, times_Vmin)
    for t in times_Vmax
        vline!(p, [t], line=:dash, color=:red, label=false)
    end
    for t in times_Vmin
        vline!(p, [t], line=:dot, color=:blue, label=false)
    end
end

v_control_markers!(p2, times_Vmax, times_Vmin)

# Constant control at Vmax

# Already defined above, skip duplicate definitions

# Constant control at Vmin

# Already defined above, skip duplicate definitions

plot(tsteps, C_num, label="Numerical Optimal V(t)", lw=2)
plot!(tsteps, C_const, label="Constant V(t) = $(round((Vmin+Vmax)/2, digits=2))", lw=2, ls=:dot)
plot!(tsteps, C_Vmax, label="Constant V(t) = Vmax = $Vmax", lw=2, ls=:dashdot)
plot!(tsteps, C_Vmin, label="Constant V(t) = Vmin = $Vmin", lw=2, ls=:dashdot)
xlabel!("Time")
ylabel!("Concentration C")
title!("Comparison of C concentration under different controls")


println("\n======================")
println("Summary of Max C for Each Control Strategy")
println("======================")
println("Numerical Optimal V(t):      ", round(maxC_num, digits=4))
println("Constant V(t) = ", round((Vmin + Vmax)/2, digits=4), ":       ", round(maxC_const, digits=4))
println("Constant V(t) = Vmax = ", Vmax, ":    ", round(maxC_Vmax, digits=4))
println("Constant V(t) = Vmin = ", Vmin, ":    ", round(maxC_Vmin, digits=4))
println("======================\n")






# --- Parameter search for Model 2 ---
"""
Run the main simulation for a given parameter set and return max C for Vmax, Vmin, Vmean, and optimal.
"""
function run_main_simulation_for_params(params)
    k1, k2, k3, Kd, D = params
    Vmin = 0.1; Vmax = 100.0
    ϵ = 1e-8
    n0 = [200.0, 0.0, 0.0]
    λT = [0.0, 0.0, 1.0]
    tspan = (0.0, 3.0)
    tsteps = range(tspan[1], tspan[2], length=300)
    rate1(A, C, V) = V * k1 * (A / (V + ϵ))  # No inhibition by C
    rate2(B, V, D) = V * k2 * (B / (V + ϵ)) / (1 + (D / (V + ϵ)) / Kd)
    rate3(C, V) = V * k3 * (C / (V + ϵ))
    function state_dynamics!(du, n, p, t, λ, V, D)
        A, B, C = n
        r1 = rate1(A, C, V)
        r2 = rate2(B, V, D)
        r3 = rate3(C, V)
        du[1] = -r1
        du[2] = r1 - r2
        du[3] = r2 - r3
    end
    function H(n, λ, V, D)
        A, B, C = n
        L1, L2, L3 = λ
        r1 = rate1(A, C, V)
        r2 = rate2(B, V, D)
        r3 = rate3(C, V)
        return L1 * (-r1) + L2 * (r1 - r2) + L3 * (r2 - r3)
    end
    function adjoint_dynamics!(dl, λ, n, V, D)
        dl .= -ForwardDiff.gradient(l -> H(n, l, V, D), λ)
    end
    function optimal_V(n, λ)
        obj(V) = -H(n, λ, V, D)
        res = optimize(obj, Vmin, Vmax, Brent())
        Vopt = Optim.minimizer(res)
        return clamp(Vopt, Vmin, Vmax)
    end
    function control_Vmax(n, λ) Vmax end
    function control_Vmin(n, λ) Vmin end
    control_Vmean(n, λ) = (Vmin + Vmax) / 2
    function forward_backward_sweep_local(n0, λT, tspan, tsteps; control_update=optimal_V)
        V_vals = fill((Vmin + Vmax) / 2, length(tsteps))
        sol_n, sol_λ = nothing, nothing
        for sweep in 1:20
            function forward_ode!(du, u, p, t)
                idx = findfirst(x -> x ≥ t, tsteps)
                V = V_vals[clamp(idx, 1, length(V_vals))]
                λ_dummy = zeros(3)
                state_dynamics!(du, u, p, t, λ_dummy, V, D)
            end
            prob_fwd = ODEProblem(forward_ode!, n0, tspan)
            sol_n = solve(prob_fwd, Tsit5(), saveat=tsteps)
            function backward_ode!(dl, l, p, t)
                idx = findfirst(x -> x ≥ t, reverse(tsteps))
                n = sol_n(tsteps[end - idx + 1])
                V = V_vals[end - idx + 1]
                adjoint_dynamics!(dl, l, n, V, D)
            end
            prob_bwd = ODEProblem(backward_ode!, λT, (tspan[2], tspan[1]))
            sol_λ = solve(prob_bwd, Tsit5(), saveat=reverse(tsteps))
            for (i, t) in enumerate(tsteps)
                n_t = sol_n(t)
                λ_t = sol_λ(tspan[2] - t)
                V_vals[i] = control_update(n_t, λ_t)
            end
        end
        return sol_n, sol_λ, V_vals
    end
    sol_n_num, sol_λ_num, V_num = forward_backward_sweep_local(n0, λT, tspan, tsteps; control_update=optimal_V)
    C_num = [sol_n_num(t)[3] for t in tsteps]
    maxC_num = maximum(C_num)
    sol_n_Vmax, _, _ = forward_backward_sweep_local(n0, λT, tspan, tsteps; control_update=control_Vmax)
    sol_n_Vmin, _, _ = forward_backward_sweep_local(n0, λT, tspan, tsteps; control_update=control_Vmin)
    sol_n_Vmean, _, _ = forward_backward_sweep_local(n0, λT, tspan, tsteps; control_update=control_Vmean)
    C_Vmax = [sol_n_Vmax(t)[3] for t in tsteps]
    C_Vmin = [sol_n_Vmin(t)[3] for t in tsteps]
    C_Vmean = [sol_n_Vmean(t)[3] for t in tsteps]
    maxC_Vmax = maximum(C_Vmax)
    maxC_Vmin = maximum(C_Vmin)
    maxC_Vmean = maximum(C_Vmean)
    return maxC_num, maxC_Vmax, maxC_Vmin, maxC_Vmean
end
function simulate_and_compare_model2(params)
    k1, k2, k3, Kd, D = params
    Vmin = 0.1; Vmax = 100.0
    n0 = [200.0, 0.0, 0.0]  # match main script
    λT = [0.0, 0.0, 1.0]
    tspan = (0.0, 3.0)
    tsteps = range(tspan[1], tspan[2], length=300)
    # Use the same rate definitions as main script (with V factor)
    function rate1(A, C, V) V * k1 * (A / (V + 1e-8))  end
    function rate2(B, V, D) V * k2 * (B / (V + 1e-8)) / (1 + (D / (V + 1e-8)) / Kd) end
    function rate3(C, V) V * k3 * (C / (V + 1e-8)) end
    function state_dynamics!(du, n, p, t, λ, V, D)
        A, B, C = n
        r1 = rate1(A, C, V)
        r2 = rate2(B, V, D)
        r3 = rate3(C, V)
        du[1] = -r1
        du[2] = r1 - r2
        du[3] = r2 - r3
    end
    function H(n, λ, V, D)
        A, B, C = n
        L1, L2, L3 = λ
        r1 = rate1(A, C, V)
        r2 = rate2(B, V, D)
        r3 = rate3(C, V)
        return L1 * (-r1) + L2 * (r1 - r2) + L3 * (r2 - r3)
    end
    function adjoint_dynamics!(dl, λ, n, V, D)
        dl .= -ForwardDiff.gradient(l -> H(n, l, V, D), λ)
    end
    function optimal_V(n, λ)
        obj(V) = -H(n, λ, V, D)
        res = optimize(obj, Vmin, Vmax, Brent())
        Vopt = Optim.minimizer(res)
        return clamp(Vopt, Vmin, Vmax)
    end
    function control_Vmax(n, λ) Vmax end
    function control_Vmin(n, λ) Vmin end
    control_Vmean(n, λ) = (Vmin + Vmax) / 2
    function forward_backward_sweep_local(n0, λT, tspan, tsteps; control_update=optimal_V)
        V_vals = fill((Vmin + Vmax) / 2, length(tsteps))
        sol_n, sol_λ = nothing, nothing
        for sweep in 1:40
            function forward_ode!(du, u, p, t)
                idx = findfirst(x -> x ≥ t, tsteps)
                V = V_vals[clamp(idx, 1, length(V_vals))]
                λ_dummy = zeros(3)
                state_dynamics!(du, u, p, t, λ_dummy, V, D)
            end
            prob_fwd = ODEProblem(forward_ode!, n0, tspan)
            sol_n = solve(prob_fwd, Tsit5(), saveat=tsteps)
            function backward_ode!(dl, l, p, t)
                idx = findfirst(x -> x ≥ t, reverse(tsteps))
                n = sol_n(tsteps[end - idx + 1])
                V = V_vals[end - idx + 1]
                adjoint_dynamics!(dl, l, n, V, D)
            end
            prob_bwd = ODEProblem(backward_ode!, λT, (tspan[2], tspan[1]))
            sol_λ = solve(prob_bwd, Tsit5(), saveat=reverse(tsteps))
            for (i, t) in enumerate(tsteps)
                n_t = sol_n(t)
                λ_t = sol_λ(tspan[2] - t)
                V_vals[i] = control_update(n_t, λ_t)
            end
        end
        return sol_n, sol_λ, V_vals
    end
    sol_n_num, sol_λ_num, V_num = forward_backward_sweep_local(n0, λT, tspan, tsteps; control_update=optimal_V)
    C_num = [sol_n_num(t)[3] for t in tsteps]
    maxC_num = maximum(C_num)
    sol_n_Vmax, _, _ = forward_backward_sweep_local(n0, λT, tspan, tsteps; control_update=control_Vmax)
    sol_n_Vmin, _, _ = forward_backward_sweep_local(n0, λT, tspan, tsteps; control_update=control_Vmin)
    sol_n_Vmean, _, _ = forward_backward_sweep_local(n0, λT, tspan, tsteps; control_update=control_Vmean)
    C_Vmax = [sol_n_Vmax(t)[3] for t in tsteps]
    C_Vmin = [sol_n_Vmin(t)[3] for t in tsteps]
    C_Vmean = [sol_n_Vmean(t)[3] for t in tsteps]
    maxC_Vmax = maximum(C_Vmax)
    maxC_Vmin = maximum(C_Vmin)
    maxC_Vmean = maximum(C_Vmean)
    maxC_const_all = maximum([maxC_Vmax, maxC_Vmin, maxC_Vmean])
    # Ensure all returned vectors are of length 300
    if length(C_num) != 300 || length(C_Vmax) != 300 || length(C_Vmin) != 300 || length(C_Vmean) != 300
        error("ODE solution did not return 300 points. C_num: $(length(C_num)), C_Vmax: $(length(C_Vmax)), C_Vmin: $(length(C_Vmin)), C_Vmean: $(length(C_Vmean))")
    end
    return maxC_num - maxC_const_all, maxC_num, maxC_Vmax, maxC_Vmin, maxC_Vmean, params
end

println("\n--- Model 2 Parameter Search (diff > 0.5) ---")
best2 = (-Inf, 0, 0, 0, 0, zeros(6))
found = false
for trial in 1:500
    k1 = 10^(rand(-2:0.1:2.0))
    k2 = 10^(rand(-2:0.1:2.0))
    k3 = 10^(rand(-2:0.1:2.0))
    Kd = 10^(rand(-2:0.1:2.0))
    D = 10^(rand(0.1:0.1:10.0))
    diff, maxC_num, maxC_Vmax, maxC_Vmin, maxC_Vmean, params = simulate_and_compare_model2([k1, k2, k3, Ki, Kd, D])
    if diff > 0.5 && isfinite(diff) && isfinite(maxC_num) && isfinite(maxC_Vmax) && isfinite(maxC_Vmin) && isfinite(maxC_Vmean)
        println("Params: k1=", params[1], ", k2=", params[2], ", k3=", params[3], ", Ki=", params[4], ", Kd=", params[5], ", D=", params[6])
        println("Numerical optimal max C: ", round(maxC_num, digits=4))
        println("Constant Vmax max C: ", round(maxC_Vmax, digits=4))
        println("Constant Vmin max C: ", round(maxC_Vmin, digits=4))
        println("Constant Vmean max C: ", round(maxC_Vmean, digits=4))
        println("Difference (numerical - best constant): ", round(diff, digits=4))
        if diff > best2[1]
            best2 = (diff, maxC_num, maxC_Vmax, maxC_Vmin, maxC_Vmean, params)
        end
        found = true
    end
end
if found
    println("\nBest found (diff > 0.5):")
    println("Difference: ", best2[1])
    println("Numerical optimal max C (parameter search): ", best2[2])
    println("Vmax (parameter search): ", best2[3], ", Vmin: ", best2[4], ", Vmean: ", best2[5])
    println("Params: k1=", best2[6][1], ", k2=", best2[6][2], ", k3=", best2[6][3], ", Ki=", best2[6][4], ", Kd=", best2[6][5], ", D=", best2[6][6])
    # Now run main simulation for these params and compare
    maxC_num_main, maxC_Vmax_main, maxC_Vmin_main, maxC_Vmean_main = run_main_simulation_for_params(best2[6])
    println("\n--- Main simulation with best parameter set ---")
    println("Numerical optimal max C (main): ", round(maxC_num_main, digits=4))
    println("Vmax (main): ", round(maxC_Vmax_main, digits=4), ", Vmin: ", round(maxC_Vmin_main, digits=4), ", Vmean: ", round(maxC_Vmean_main, digits=4))
    println("\nCompare: If these do not match, check for hidden differences in rates, ICs, or time grid.")
else
    println("\nNo parameter set found where numerical optimal C exceeds all constant controls by more than 0.5.")
end

function plot_numerical_volume(tsteps, V_vals)
    pV = plot(tsteps, V_vals, label="Numerical Optimal V(t)", lw=2, color=:black)
    xlabel!(pV, "Time")
    ylabel!(pV, "V(t)")
    title!(pV, "Numerical Optimal Control V(t)")
    display(pV)
end

# Example usage (uncomment to run):
plot_numerical_volume(tsteps, V_vals)












using Plots
tsteps = (0.0, 4.0)
# -------------------------
# Extract trajectories
# -------------------------
A_opt = [sol_n_num(t)[1] for t in tsteps]
B_opt = [sol_n_num(t)[2] for t in tsteps]
C_opt = [sol_n_num(t)[3] for t in tsteps]

C_Vmin = [sol_n_Vmin(t)[3] for t in tsteps]
C_Vmax = [sol_n_Vmax(t)[3] for t in tsteps]

# -------------------------
# Top panel: Vopt
# -------------------------
p1 = plot(
    tsteps, V_num,
    xlabel = "",
    ylabel = "V(t)",
    title = "Optimal Control Vopt(t)",
    lw = 2,
    legend = false
)

# -------------------------
# Middle panel: Species (Vopt)
# -------------------------
p2 = plot(
    tsteps, A_opt,
    label = "A",
    lw = 2
)
plot!(p2, tsteps, B_opt, lw = 2, label = "B")
plot!(p2, tsteps, C_opt, lw = 2, label = "C")

xlabel!(p2, "")
ylabel!(p2, "Concentration")
title!(p2, "Species Dynamics under Vopt")

# -------------------------
# Bottom panel: C comparison
# -------------------------
p3 = plot(
    tsteps, C_opt,
    lw = 3,
    label = "C (Vopt)"
)
plot!(p3, tsteps, C_Vmin, lw = 2, ls = :dash, label = "C (Vmin)")
plot!(p3, tsteps, C_Vmax, lw = 2, ls = :dot,  label = "C (Vmax)")

xlabel!(p3, "Time")
ylabel!(p3, "C")
title!(p3, "C(t) Comparison")

# -------------------------
# Combine into one figure
# -------------------------
plot(
    p1, p2, p3,
    layout = @layout([a; b; c]),
    size = (800, 900)
)











############################################################
# No C Inhibition – Full Working Script
############################################################
Pkg.add("ForwardDiff")
using DifferentialEquations
using Optim
using ForwardDiff
using Plots



############################################################
# Main simulation function
############################################################
function run_main_simulation_for_params(params)

    # -------------------------
    # Parameters
    # -------------------------
    k1, k2, k3, Ki, Kd, D = params
    Vmin, Vmax = 0.1, 100.0
    ϵ = 1e-8

    n0 = [200.0, 0.0, 0.0]
    λT = [0.0, 0.0, 1.0]

    tspan = (0.0, 3.0)
    tsteps = range(tspan[1], tspan[2], length=300)

    # -------------------------
    # Rates
    # -------------------------
    rate1(A, V) = V * k1 * (A / (V + ϵ))
    rate2(B, V) = V * k2 * (B / (V + ϵ)) / (1 + (D / (V + ϵ)) / Kd)
    rate3(C, V) = V * k3 * (C / (V + ϵ))

    # -------------------------
    # State dynamics
    # -------------------------
    function state_dynamics!(du, n, V)
        A, B, C = n
        r1 = rate1(A, V)
        r2 = rate2(B, V)
        r3 = rate3(C, V)
        du[1] = -r1
        du[2] = r1 - r2
        du[3] = r2 - r3
    end

    # -------------------------
    # Hamiltonian
    # -------------------------
    function H(n, λ, V)
        A, B, C = n
        L1, L2, L3 = λ
        r1 = rate1(A, V)
        r2 = rate2(B, V)
        r3 = rate3(C, V)
        return L1*(-r1) + L2*(r1 - r2) + L3*(r2 - r3)
    end

    function adjoint_dynamics!(dl, λ, n, V)
        dl .= -ForwardDiff.gradient(l -> H(n, l, V), λ)
    end

    # -------------------------
    # Controls
    # -------------------------
    function optimal_V(n, λ)
        obj(V) = -H(n, λ, V)
        res = optimize(obj, Vmin, Vmax, Brent())
        clamp(Optim.minimizer(res), Vmin, Vmax)
    end

    control_Vmin(n, λ) = Vmin
    control_Vmax(n, λ) = Vmax

    # -------------------------
    # Forward–Backward Sweep
    # -------------------------
    function forward_backward_sweep(control_update)

        Vvals = fill((Vmin + Vmax)/2, length(tsteps))
        sol_n = nothing  # <<< CRITICAL FIX

        for _ in 1:25

            # ---------- Forward ----------
            function forward!(du, u, p, t)
                i = clamp(findfirst(x -> x ≥ t, tsteps), 1, length(Vvals))
                state_dynamics!(du, u, Vvals[i])
            end

            sol_n = solve(
                ODEProblem(forward!, n0, tspan),
                Tsit5(), saveat=tsteps
            )

            # ---------- Backward ----------
            function backward!(dl, l, p, t)
                i = clamp(findfirst(x -> x ≥ t, reverse(tsteps)), 1, length(Vvals))
                n = sol_n(tsteps[end - i + 1])
                adjoint_dynamics!(dl, l, n, Vvals[end - i + 1])
            end

            sol_λ = solve(
                ODEProblem(backward!, λT, (tspan[2], tspan[1])),
                Tsit5(), saveat=reverse(tsteps)
            )

            # ---------- Control update ----------
            for (i, t) in enumerate(tsteps)
                Vvals[i] = control_update(sol_n(t), sol_λ(tspan[2] - t))
            end
        end

        return sol_n, Vvals
    end

    # -------------------------
    # Run simulations
    # -------------------------
    sol_opt, Vopt = forward_backward_sweep(optimal_V)
    sol_Vmin, _ = forward_backward_sweep(control_Vmin)
    sol_Vmax, _ = forward_backward_sweep(control_Vmax)

    return (
        t = tsteps,
        sol_opt = sol_opt,
        sol_Vmin = sol_Vmin,
        sol_Vmax = sol_Vmax,
        Vopt = Vopt
    )
end

############################################################
# Run model
############################################################
params = (
    1.6,   # k1
    1,   # k2
    1,   # k3
    1.0,   # Ki (unused)
    0.3,  # Kd
    6.7   # D
)
k1, k2, k3 = 1.6, 1 , 1
Ki = 2
Kd = 0.3
D = 6.7
results = run_main_simulation_for_params(params)

############################################################
# Extract trajectories
############################################################
t = results.t
Vopt = results.Vopt

A = [results.sol_opt(tt)[1] for tt in t]
B = [results.sol_opt(tt)[2] for tt in t]
C = [results.sol_opt(tt)[3] for tt in t]

Cmin = [results.sol_Vmin(tt)[3] for tt in t]
Cmax = [results.sol_Vmax(tt)[3] for tt in t]

############################################################
# Plot
############################################################
p1 = plot(t, Vopt, lw=2, title="Optimal Control Vopt(t)",
          ylabel="V(t)", legend=false)

p2 = plot(t, A, lw=2, label="A", title="Species under Vopt",
          ylabel="Concentration")
plot!(p2, t, B, lw=2, label="B")
plot!(p2, t, C, lw=2, label="C")

p3 = plot(t, C, lw=3, label="C (Vopt)",
          title="C Comparison", xlabel="Time", ylabel="C")
plot!(p3, t, Cmin, lw=2, ls=:dash, label="C (Vmin)")
plot!(p3, t, Cmax, lw=2, ls=:dot, label="C (Vmax)")

plot(p1, p2, p3, layout=@layout([a; b; c]), size=(850, 900))








############################################################
# Volume-Controlled A → B → C System with Inhibition on B→C
# Monte Carlo search + single-run plotting
############################################################

using Pkg
Pkg.activate(".")
using DifferentialEquations, Optim, Random, Statistics, Plots

# -------------------------
# Global settings
# -------------------------
const ϵ = 1e-9
Vmin, Vmax = 1.0, 100.0
tspan = (0.0, 500.0)

SAVEAT      = collect(range(tspan[1], tspan[2], length=1001))
SAVEAT_FAST = collect(range(tspan[1], tspan[2], length=201))

u0 = [100.0, 0.0, 0.0]   # [A, B, C]

# -------------------------
# Reaction rates (YOUR DEFINITIONS)
# -------------------------
rate1(A, V, k1) =
    V * k1 * (A / (V + ϵ))

rate2(B, V, D, k2, Kd) =
    V * k2 * (B / (V + ϵ)) / (1 + (D / (V + ϵ)) / Kd)

rate3(C, V, k3) =
    V * k3 * (C / (V + ϵ))

# -------------------------
# ODE model
# p = (k1, k2, k3, D, Kd)
# -------------------------
function model_ABC!(du, u, p, t, V)
    A, B, C = u
    k1, k2, k3, D, Kd = p

    r1 = rate1(A, V, k1)
    r2 = rate2(B, V, D, k2, Kd)
    r3 = rate3(C, V, k3)

    du[1] = -r1
    du[2] =  r1 - r2
    du[3] =  r2 - r3
end

# -------------------------
# Volume controls
# -------------------------
function sinusoidal_V(t, A, ω, ϕ)
    Vmean = (Vmin + Vmax)/2
    clamp(Vmean + A*sin(ω*t + ϕ), Vmin, Vmax)
end

# -------------------------
# Simulation wrappers
# -------------------------
function simulate_constant(p, V; saveat=SAVEAT_FAST)
    f!(du,u,p_,t) = model_ABC!(du,u,p_,t,V)
    prob = ODEProblem(f!, copy(u0), tspan, p)
    solve(prob, TRBDF2(); saveat, abstol=1e-6, reltol=1e-6)
end

function simulate_sinusoidal(p, A, ω, ϕ; saveat=SAVEAT_FAST)
    f!(du,u,p_,t) = model_ABC!(du,u,p_,t,
                              sinusoidal_V(t,A,ω,ϕ))
    prob = ODEProblem(f!, copy(u0), tspan, p)
    solve(prob, TRBDF2(); saveat, abstol=1e-6, reltol=1e-6)
end

# -------------------------
# Average C(t)
# -------------------------
function avg_C(sol)
    t = sol.t
    C = [u[3] for u in sol.u]
    sum(0.5*(C[i]+C[i+1])*(t[i+1]-t[i])
        for i in 1:length(t)-1) / (t[end]-t[1])
end

# -------------------------
# Optimize sinusoidal control
# -------------------------
function find_optimal_sinusoidal(params)
    obj(x) = begin
        sol = simulate_sinusoidal(params, x[1], x[2], x[3])
        sol.retcode == :Success ? -avg_C(sol) : 1e9
    end

    lower = [0.0, 0.005, 0.0]
    upper = [(Vmax-Vmin)/2, 1.0, 2π]
    x0    = [(Vmax-Vmin)/4, 0.05, 0.0]

    res = optimize(obj, lower, upper, x0,
                   Fminbox(NelderMead()),
                   Optim.Options(iterations=600))

    res.minimizer
end

# -------------------------
# Random parameter generator (NOW INCLUDES Kd)
# -------------------------
function random_parameters_ABC()
    k1 = 10^(rand(-2.0:0.05:0.5))     # 0.01 – 3
    k2 = 10^(rand(-2.0:0.05:0.5))
    k3 = 10^(rand(-2.0:0.05:0.5))
    D  = 10^(rand(-3.0:0.05:-0.3))    # inhibitor level
    Kd = 10^(rand(-3.0:0.05:1.0))     # inhibition strength
    return (k1, k2, k3, D, Kd)
end

# -------------------------
# Monte Carlo search
# -------------------------
function monte_carlo_search(n_trials=20; gain=0.5)
    successes = []

    for i in 1:n_trials
        println("\n🧪 Trial $i / $n_trials")
        params = random_parameters_ABC()

        try
            Aopt, ωopt, ϕopt = find_optimal_sinusoidal(params)

            Copt  = avg_C(simulate_sinusoidal(params, Aopt, ωopt, ϕopt))
            Cmin  = avg_C(simulate_constant(params, Vmin))
            Cmax  = avg_C(simulate_constant(params, Vmax))
            Cmean = avg_C(simulate_constant(params, (Vmin+Vmax)/2))

            if Copt > Cmin+gain &&
               Copt > Cmax+gain &&
               Copt > Cmean+gain

                println("✅ SUCCESS")
                push!(successes, (
                    params=params,
                    Copt=Copt,
                    Cmin=Cmin,
                    Cmax=Cmax,
                    Cmean=Cmean,
                    Vopt=(Aopt, ωopt, ϕopt)
                ))
            else
                println("⚪ No dominance")
            end
        catch e
            println("❌ Failed trial: ", e)
        end
    end
    successes
end

# -------------------------
# Run Monte Carlo
# -------------------------
Random.seed!(11)
println("\nStarting Monte Carlo search...")
successes = monte_carlo_search(20)

println("\n=== SUMMARY ===")
println("Found $(length(successes)) winning parameter sets.")

# -------------------------
# Plot example if success exists
# -------------------------
if !isempty(successes)
    s = successes[1]
    (Aopt, ωopt, ϕopt) = s.Vopt

    sol_opt  = simulate_sinusoidal(s.params, Aopt, ωopt, ϕopt; saveat=SAVEAT)
    sol_min  = simulate_constant(s.params, Vmin; saveat=SAVEAT)
    sol_max  = simulate_constant(s.params, Vmax; saveat=SAVEAT)

    Vopt_t = [sinusoidal_V(t, Aopt, ωopt, ϕopt) for t in SAVEAT]

    p1 = plot(SAVEAT, Vopt_t, lw=3, label="Vopt",
              xlabel="Time", ylabel="Volume")

    p2 = plot(SAVEAT, [u[1] for u in sol_opt.u], label="A")
    plot!(p2, SAVEAT, [u[2] for u in sol_opt.u], label="B")
    plot!(p2, SAVEAT, [u[3] for u in sol_opt.u], lw=3, label="C")
    xlabel!("Time"); ylabel!("Concentration")

    p3 = plot(SAVEAT, [u[3] for u in sol_opt.u], lw=3, label="Vopt")
    plot!(p3, SAVEAT, [u[3] for u in sol_min.u], ls=:dash, label="Vmin")
    plot!(p3, SAVEAT, [u[3] for u in sol_max.u], ls=:dot, label="Vmax")
    xlabel!("Time"); ylabel!("C(t)")

    plot(p1, p2, p3, layout=(3,1), size=(800,900))
end


0.5011872336272722, 0.03162277660168379, 0.1, 0.0012589254117941675, 0.0446683592150963



############################################################
# ========== SINGLE RUN: USER PARAMETERS ==================
############################################################

# >>>>>>> EDIT ONLY THIS <<<<<<<
params = (
0.5011872336272722, 0.03162277660168379, 0.1, 0.0012589254117941675, 0.0446683592150963
              # Kd
)

############################################################
# Optimize sinusoidal volume
############################################################

Aopt, ωopt, ϕopt = find_optimal_sinusoidal(params)

println("\nOptimal sinusoidal control:")
println("A = $Aopt, ω = $ωopt, ϕ = $ϕopt")

############################################################
# Run simulations
############################################################

sol_opt  = simulate_sinusoidal(params, Aopt, ωopt, ϕopt; saveat=SAVEAT)
sol_min  = simulate_constant(params, Vmin; saveat=SAVEAT)
sol_max  = simulate_constant(params, Vmax; saveat=SAVEAT)
sol_mean = simulate_constant(params, (Vmin+Vmax)/2; saveat=SAVEAT)

# Optimal volume trace
Vopt_t = [sinusoidal_V(t, Aopt, ωopt, ϕopt) for t in SAVEAT]

############################################################
# ======================= PLOTS ============================
############################################################

# ---- Plot 1: Optimal Volume ----
p1 = plot(
    SAVEAT, Vopt_t,
    lw=3,
    xlabel="Time",
    ylabel="Volume V(t)",
    title="Optimal Volume Control",
    label="Vopt"
)

# ---- Plot 2: A, B, C under Vopt ----
A = [u[1] for u in sol_opt.u]
B = [u[2] for u in sol_opt.u]
C = [u[3] for u in sol_opt.u]

p2 = plot(SAVEAT, A, lw=2, label="A")
plot!(p2, SAVEAT, B, lw=2, label="B")
plot!(p2, SAVEAT, C, lw=3, label="C")
xlabel!("Time")
ylabel!("Concentration")
title!("Species under Optimal Volume")

# ---- Plot 3: C(t) comparison ----
p3 = plot(
    SAVEAT, [u[3] for u in sol_opt.u],
    lw=3,
    label="Vopt"
)
plot!(p3, SAVEAT, [u[3] for u in sol_min.u],  lw=2, ls=:dash,    label="Vmin")
plot!(p3, SAVEAT, [u[3] for u in sol_mean.u], lw=2, ls=:dot,     label="Vmean")
plot!(p3, SAVEAT, [u[3] for u in sol_max.u],  lw=2, ls=:dashdot, label="Vmax")
xlabel!("Time")
ylabel!("C(t)")
title!("C(t) Comparison")

# ---- Display all ----
plot(p1, p2, p3, layout=(3,1), size=(850,900))




############################################################



using Pkg
Pkg.activate(".")
using DifferentialEquations, Optim, Random, Plots, Statistics

# Parameters
k1, k2, k3 = 1.6, 1 , 1
Kd = 0.3
D = 6.7
Vmin, Vmax = 1.0, 100.0
ϵ = 1e-8  # small value to avoid division by zero

# Rates with stabilizers
# Restore C inhibition to A->B rate
rate1(A, C, V) = V * k1 * (A / (V + ϵ))
rate2(B, V, D) = V * k2 * (B / (V + ϵ)) / (1 + (D / (V + ϵ)) / Kd)
rate3(C, V) = V * k3 * (C / (V + ϵ))

# System dynamics: n = [A, B, C]
function state_dynamics!(du, n, p, t, λ, V, D)
    A, B, C = n
    r1 = rate1(A, C, V)
    r2 = rate2(B, V, D)
    r3 = rate3(C, V)
    du[1] = -r1
    du[2] = r1 - r2
    du[3] = r2 - r3
end

# Hamiltonian
function H(n, λ, V, D)
    A, B, C = n
    L1, L2, L3 = λ
    r1 = rate1(A, C, V)
    r2 = rate2(B, V, D)
    r3 = rate3(C, V)
    return L1 * (-r1) + L2 * (r1 - r2) + L3 * (r2 - r3)
end

# Adjoint dynamics
function adjoint_dynamics!(dl, λ, n, V, D)
    dl .= -ForwardDiff.gradient(n -> H(n, λ, V, D), n)
end

# Pontryagin's Maximum Principle: maximize Hamiltonian w.r.t. V
function optimal_V(n, λ)
    obj(V) = -H(n, λ, V, D)  # minimize -H is same as maximize H
    res = optimize(obj, Vmin, Vmax, Brent())
    Vopt = Optim.minimizer(res)
    return clamp(Vopt, Vmin, Vmax)
end

# Forward-backward sweep
function forward_backward_sweep(n0, λT, tspan, tsteps; control_update=optimal_V, iters = 100)
    V_vals = fill((Vmin + Vmax) / 2, length(tsteps))
    sol_n, sol_λ = nothing, nothing
    for sweep in 1:iters
        println("Sweep $sweep")
        λ_dummy = zeros(3)
        # Forward ODE
        function forward_ode!(du, u, p, t)
            idx = findfirst(x -> x ≥ t, tsteps)
            state_dynamics!(du, u, p, t, λ_dummy, V_vals[idx], D)
        end
        prob_fwd = ODEProblem(forward_ode!, n0, tspan)
        sol_n = solve(prob_fwd, Tsit5(), saveat=tsteps)

        # Backward ODE (adjoint)
        function backward_ode!(dl, l, p, t)
            idx = findfirst(x -> x ≥ t, tsteps)
            adjoint_dynamics!(dl, l, sol_n(t), V_vals[idx], D)
        end
        prob_bwd = ODEProblem(backward_ode!, λT, (tspan[2], tspan[1]))
        sol_λ = solve(prob_bwd, Tsit5(), saveat=reverse(tsteps))
        # Update control using optimal V(t)
        for (i, t) in enumerate(tsteps)
            n_t = sol_n(t)
            λ_t = sol_λ(t)
            V_vals[i] = control_update(n_t, λ_t)
        end
    end
    return sol_n, sol_λ, V_vals
end

# Initial conditions
n0 = [200.0, 0.0, 0.0]  # [A, B, C]
λT = [0.0, 0.0, 1.0]   # final adjoint state: sensitivity to C

tspan = (0.0, 4.0)
tsteps = range(tspan[1], tspan[2], length=300)

# Run forward-backward sweep for optimal control
control_Vmax = Returns(Vmax)
control_Vmin = Returns(Vmin)
control_Vmean = Returns((Vmin + Vmax) / 2)

function generate_solutions(n0, λT, tspan, tsteps)
    sol_n, sol_λ, V_vals = forward_backward_sweep(n0, λT, tspan, tsteps);
    sol_n_Vmax, _, _ = forward_backward_sweep(n0, λT, tspan, tsteps; control_update=control_Vmax);
    sol_n_Vmin, _, _ = forward_backward_sweep(n0, λT, tspan, tsteps; control_update=control_Vmin);
    sol_n_Vmean, _, _ = forward_backward_sweep(n0, λT, tspan, tsteps; control_update=control_Vmean);
    
    (; opt = sol_n, max = sol_n_Vmax, min = sol_n_Vmin, mean = sol_n_Vmean)
end

# Plot all C curves together
function plot_concentration_curves(sols; idxs = 3)
    pC = plot(sols.opt, idxs = idxs, label="Optimal V(t)", lw=2)
    plot!(sols.max, idxs = idxs, label="Constant Vmax", lw=2, ls=:dash)
    plot!(sols.min, idxs = idxs, label="Constant Vmin", lw=2, ls=:dot)
    plot!(sols.mean, idxs = idxs, label="Constant Vmean", lw=2, ls=:dashdot)
    xlabel!("Time")
    ylabel!("Concentration C")
    title!("C(t) under different controls")
end

# Plot V(t) for optimal
function plot_volume_curve()
    # This function will be completed in the main block below
    return nothing
end

function plot_vc()
    # This function will be completed in the main block below
    return nothing
# === MAIN BLOCK ===
n0 = [200.0, 0.0, 0.0]
λT = [0.0, 0.0, 1.0]
tspan = (0.0, 4.0)
tsteps = range(tspan[1], tspan[2], length=300)

sols = generate_solutions(n0, λT, tspan, tsteps)

# Extract C(t) for each control
C_opt = [sols.opt(t)[3] for t in tsteps]
C_Vmax = [sols.max(t)[3] for t in tsteps]
C_Vmin = [sols.min(t)[3] for t in tsteps]
C_Vmean = [sols.mean(t)[3] for t in tsteps]

# Extract V(t) for optimal control
_, _, V_vals = forward_backward_sweep(n0, λT, tspan, tsteps)

# Print max C for each control
println("Max C (Optimal): ", maximum(C_opt))
println("Max C (Vmax): ", maximum(C_Vmax))
println("Max C (Vmin): ", maximum(C_Vmin))
println("Max C (Vmean): ", maximum(C_Vmean))

# Plot C curves
plot_concentration_curves(sols)

# Plot V(t) for optimal control
plot(tsteps, V_vals, label="Optimal V(t)", lw=2)
xlabel!("Time")
ylabel!("Volume V(t)")
title!("Optimal Control V(t)")
end

# Plot all together

# Print max C for each control
println("Max C (Optimal): ", maximum(C_opt))
println("Max C (Vmax): ", maximum(C_Vmax))
println("Max C (Vmin): ", maximum(C_Vmin))
println("Max C (Vmean): ", maximum(C_Vmean))



# Detect times when Vmax or Vmin used in numerical optimal control

# If V_num, C_num, C_const are not defined, use V_vals, C_opt, C_Vmean as fallback
# Ensure all required variables are defined in global scope
sols = generate_solutions(n0, λT, tspan, tsteps)
C_opt = [sols.opt(t)[3] for t in tsteps]
C_Vmax = [sols.max(t)[3] for t in tsteps]
C_Vmin = [sols.min(t)[3] for t in tsteps]
C_Vmean = [sols.mean(t)[3] for t in tsteps]
_, _, V_vals = forward_backward_sweep(n0, λT, tspan, tsteps)
V_num = V_vals
C_num = C_opt
C_const = C_Vmean
times_Vmax = [tsteps[i] for i in 1:length(V_num) if isapprox(V_num[i], Vmax; atol=1e-4)]
times_Vmin = [tsteps[i] for i in 1:length(V_num) if isapprox(V_num[i], Vmin; atol=1e-4)]

# Compute and print max concentrations for each control
maxC_num = maximum(C_num)
maxC_const = maximum(C_const)
maxC_Vmax = maximum(C_Vmax)
maxC_Vmin = maximum(C_Vmin)
println("maxC_num = ", maxC_num)
println("maxC_const = ", maxC_const)
println("maxC_Vmax = ", maxC_Vmax)
println("maxC_Vmin = ", maxC_Vmin)

println("Times when Vmax is used in numerical optimal control:")
println(times_Vmax)

println("Times when Vmin is used in numerical optimal control:")
println(times_Vmin)

# Find max C from all controls (numerical optimal and constant)

# Compute maxC_Vmax and maxC_Vmin for summary printout
maxC_num = maximum(C_num)
maxC_const = maximum(C_const)
maxC_Vmax = maximum(C_Vmax)
maxC_Vmin = maximum(C_Vmin)

println("Max concentration C (Numerical Optimal Control): ", maxC_num)
println("Max concentration C (Constant Control): ", maxC_const)

Pkg.add("RecipesBase")
using RecipesBase

function v_control_markers!(p, times_Vmax, times_Vmin)
    for t in times_Vmax
        vline!(p, [t], line=:dash, color=:red, label=false)
    end
    for t in times_Vmin
        vline!(p, [t], line=:dot, color=:blue, label=false)
    end
end

v_control_markers!(p2, times_Vmax, times_Vmin)

# Constant control at Vmax

# Already defined above, skip duplicate definitions

# Constant control at Vmin

# Already defined above, skip duplicate definitions

plot(tsteps, C_num, label="Numerical Optimal V(t)", lw=2)
plot!(tsteps, C_const, label="Constant V(t) = $(round((Vmin+Vmax)/2, digits=2))", lw=2, ls=:dot)
plot!(tsteps, C_Vmax, label="Constant V(t) = Vmax = $Vmax", lw=2, ls=:dashdot)
plot!(tsteps, C_Vmin, label="Constant V(t) = Vmin = $Vmin", lw=2, ls=:dashdot)
xlabel!("Time")
ylabel!("Concentration C")
title!("Comparison of C concentration under different controls")


println("\n======================")
println("Summary of Max C for Each Control Strategy")
println("======================")
println("Numerical Optimal V(t):      ", round(maxC_num, digits=4))
println("Constant V(t) = ", round((Vmin + Vmax)/2, digits=4), ":       ", round(maxC_const, digits=4))
println("Constant V(t) = Vmax = ", Vmax, ":    ", round(maxC_Vmax, digits=4))
println("Constant V(t) = Vmin = ", Vmin, ":    ", round(maxC_Vmin, digits=4))
println("======================\n")




