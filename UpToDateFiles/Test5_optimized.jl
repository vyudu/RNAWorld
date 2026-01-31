
# Optimize_Average_RNA_Model_fixed_optimized_v2.jl
using Pkg
Pkg.activate(".")
using DifferentialEquations, Optim, Random, Plots, Statistics, BlackBoxOptim
Pkg.add("BlackBoxOptim")
# -------------------------
# Global constants & params
# -------------------------
const eps_div = 1e-9

Vmin, Vmax = 1, 100.0
# increase grid density for accurate averages
tspan = (0.0, 1000.0)   # you already set 700; change to 1000 if needed
t_steps = range(tspan[1], tspan[2], length=1001)  # denser accurate grid
const SAVEAT = collect(t_steps)

# coarse save grid for fast objective evaluations during optimization
const SAVEAT_FAST = collect(range(tspan[1], tspan[2], length=101))  # 101 -> better coarse capture
n0 = [100.0, 50.0, 10.0, 50.0, 100.]   # initial conditions: [α, Cp, αCp, CppC, RNA]

# Example starting params (k1,k2,k3,k4,k5,c6,Kd,D1) - used only for example plots later
params_example = (0.018620871366628676, 0.13489628825916536, 0.03890451449942807, 0.016595869074375606, 0.03548133892335755, 17.378008287493753, 0.20417379446695297, 0.13489628825916536)

# -----------------------
# Reaction model 
# -------------------------
function rates(n::AbstractVector, V, p)
    α, Cp, αCp, CppC, RNA = n
    k1,k2,k3,k4,k5,c6,Kd,D1 = p
    Vsafe = max(float(V), eps_div)

    r1 = k1 * (αCp^2) / (Vsafe^2)
    r2 = k2 * (α * CppC) / (Vsafe^2)
    r3 = k3 * CppC / Vsafe
    r4 = k4 * (α * Cp) / (Vsafe^2)
    r5 = k5 * αCp / Vsafe
    r6 = c6 * (CppC / Vsafe) / (1 + (Cp + αCp) / (Kd * Vsafe))
    r7 = D1 / Vsafe
    return r1,r2,r3,r4,r5,r6,r7
end

function model!(du, n, p, t, V)
    r1,r2,r3,r4,r5,r6,r7 = rates(n, V, p)
    du[1] = r1 + r5 - r2 - r4
    du[2] = r3 + r5 - r4 + r7
    du[3] = 2r2 - 2r1 + r3 + r4 - r5 + r6
    du[4] = r1 - r2 - r3 - r6
    du[5] = r6 - r7
end

# -------------------------
# Controls
# -------------------------
# sinusoidal control function
function sinusoidal_V(t, A, ω, ϕ)
    Vmean = (Vmax + Vmin) / 2
    raw = Vmean + A * sin(ω * t + ϕ)
    return clamp(raw, Vmin, Vmax)
end

# -------------------------
# Simulation wrappers (consistent save grid)
# -------------------------
# simulate with sinusoidal control (A, ω, ϕ)
function simulate_sinusoidal(p::NTuple, n0, tspan, A, ω, ϕ; saveat = SAVEAT, abstol=1e-8, reltol=1e-8)
    f!(du, u, p_, t) = model!(du, u, p_, t, sinusoidal_V(t, A, ω, ϕ))
    prob = ODEProblem(f!, copy(n0), tspan, p)
    # use stiff solver for biochemical kinetics
    sol = solve(prob, TRBDF2();
    abstol=1e-5, reltol=1e-5,
    saveat=SAVEAT_FAST,
    maxiters=1e7)

    return sol
end

# simulate with constant V
function simulate_constant(p::NTuple, n0, tspan, Vconst; saveat = SAVEAT, abstol=1e-8, reltol=1e-8)
    f!(du, u, p_, t) = model!(du, u, p_, t, Vconst)
    prob = ODEProblem(f!, copy(n0), tspan, p)
    sol = solve(prob, TRBDF2();
    abstol=1e-5, reltol=1e-5,
    saveat=SAVEAT_FAST,
    maxiters=1e7)

    return sol
end

# -------------------------
# Fast average-RNA integrator (allocation-light trapezoid on saved grid)
# -------------------------
function avg_RNA_from_sol(sol)
    tt = sol.t
    s = 0.0
    N = length(tt)
    if N < 2
        return 0.0
    end
    for i in 1:(N-1)
        y1 = sol.u[i][5]
        y2 = sol.u[i+1][5]
        dt = tt[i+1] - tt[i]
        s += 0.5 * (y1 + y2) * dt
    end
    return s / (tt[end] - tt[1])
end

# -------------------------
# Sinusoidal optimizer (A, ω, ϕ)
# -------------------------
function find_optimal_sinusoidal(params; verbose=false)
    obj(x) = begin
        A, ω, ϕ = x
        sol = simulate_sinusoidal(params, n0, tspan, A, ω, ϕ;
                                  saveat=SAVEAT_FAST, abstol=1e-6, reltol=1e-6)
        if sol.retcode != :Success
            return 1e6
        end
        avg_RNA = avg_RNA_from_sol(sol)
        return -avg_RNA
    end

    lower = [0.0, 0.005, 0.0]   # allow slightly lower ω to capture slow forcing
    upper = [(Vmax - Vmin)/2, 1.0, 2*pi]
    x0 = [ (Vmax - Vmin)/4, 0.05, 0.0 ]

    # increase iterations significantly (good balance: 1500)
    res = optimize(obj, lower, upper, x0,
                   Fminbox(NelderMead()),
                   Optim.Options(iterations=1500, show_trace=verbose))
    A_opt, ω_opt, ϕ_opt = res.minimizer

    # Re-evaluate best candidate thoroughly
    sol_refined = simulate_sinusoidal(params, n0, tspan, A_opt, ω_opt, ϕ_opt;
                                      saveat=SAVEAT, abstol=1e-8, reltol=1e-8)
    if sol_refined.retcode != :Success
        best_avg = -res.minimum
    else
        best_avg = avg_RNA_from_sol(sol_refined)
    end

    return (A_opt, ω_opt, ϕ_opt), best_avg
end

# -------------------------
# Random parameter generator (exponential sampling)
# -------------------------
   # --- Upstream interactions kept modest (avoid “everything becomes RNA”) ---
function random_parameters_extreme()
    # upstream very broad
    k1 = 10^(rand(-4.0:0.01:0.0))    # 0.0001 – 1
    k2 = 10^(rand(-4.0:0.01:0.0))    # 0.0001 – 1
    k3 = 10^(rand(-4.0:0.01:0.0))    # 0.0001 – 1
    k4 = 10^(rand(-4.0:0.01:0.0))    # 0.0001 – 1
    k5 = 10^(rand(-4.0:0.01:0.0))    # 0.0001 – 1

    # massive nonlinear RNA generation
    c6 = 10^(rand(2.0:0.01:5.0))     # 100 – 100,000

    # very strong inhibition
    Kd = 10^(rand(-6.0:0.01:-0.5))   # 1e-6 – 0.3

    # very small drain
    D1 = 10^(rand(-6.0:0.01:-0.5))   # 1e-6 – 0.3

    return (k1, k2, k3, k4, k5, c6, Kd, D1)
end

# -------------------------
# Monte Carlo parameter search
# -------------------------
function monte_carlo_search(n_trials::Int=50; require_gain=5.0, verbose=false)
    successes = []
    for i in 1:n_trials
        println("\n🧪 Trial $i / $n_trials")
        params_rand = random_parameters()

        try
            # 1) find best sinusoidal control for these kinetics
            (Aopt, wopt, phopt), avg_RNA_sig = find_optimal_sinusoidal(params_rand; verbose=false)

            # 2) simulate constant Vmin and Vmax on same save grid
            sol_min = simulate_constant(params_rand, n0, tspan, Vmin; saveat=SAVEAT, abstol=1e-8, reltol=1e-8)
            sol_max = simulate_constant(params_rand, n0, tspan, Vmax; saveat=SAVEAT, abstol=1e-8, reltol=1e-8)

            avg_RNA_min = avg_RNA_from_sol(sol_min)
            avg_RNA_max = avg_RNA_from_sol(sol_max)

            if avg_RNA_sig > (avg_RNA_min + require_gain) && avg_RNA_sig > (avg_RNA_max + require_gain)
                push!(successes, (
                    params = params_rand,
                    sinusoidal_avg = avg_RNA_sig,
                    Vmin_avg = avg_RNA_min,
                    Vmax_avg = avg_RNA_max,
                    Vopt_params = (Aopt, wopt, phopt)
                ))
                println("✅ Success! RNA gain ≥ $require_gain over both Vmin/Vmax.")
            else
                println("⚪ No significant gain (sig=$(round(avg_RNA_sig,digits=3)), min=$(round(avg_RNA_min,digits=3)), max=$(round(avg_RNA_max,digits=3))).")
            end
        catch e
            println("❌ Skipped trial due to: ", e)
            continue
        end
    end
    return successes
end

# -------------------------
# Run Monte Carlo (adjust n_trials small first while testing)
# -------------------------
Random.seed!(1)  # reproducible demo seed
n_trials = 50
println("\nStarting Monte Carlo search with $n_trials trials...")
successes = monte_carlo_search(n_trials; require_gain=5.0, verbose=false)

println("\n=== SUMMARY ===")
println("Found $(length(successes)) parameter sets with ≥10 RNA advantage.")
for s in successes
    println(s)
end

# -------------------------
# Example: simulate & plot the optimized sinusoidal for params_example
# -------------------------
(A_opt, ω_opt, ϕ_opt), avg_opt_example = find_optimal_sinusoidal(params_example; verbose=false)
println("\nExample optimized sinusoidal for example params:")
println("A = $(A_opt), ω = $(ω_opt), ϕ = $(ϕ_opt), avg_RNA = $(avg_opt_example)")

sol_sig = simulate_sinusoidal(params_example, n0, tspan, A_opt, ω_opt, ϕ_opt; saveat=SAVEAT, abstol=1e-8, reltol=1e-8)
sol_min = simulate_constant(params_example, n0, tspan, Vmin; saveat=SAVEAT, abstol=1e-8, reltol=1e-8)
sol_max = simulate_constant(params_example, n0, tspan, Vmax; saveat=SAVEAT, abstol=1e-8, reltol=1e-8)

avg_sig = avg_RNA_from_sol(sol_sig)
avg_min = avg_RNA_from_sol(sol_min)
avg_max = avg_RNA_from_sol(sol_max)

# Plot RNA time-courses on same axes
RNA_sig = [sol_sig.u[i][5] for i in 1:length(sol_sig.t)]
RNA_min = [sol_min.u[i][5] for i in 1:length(sol_min.t)]
RNA_max = [sol_max.u[i][5] for i in 1:length(sol_max.t)]

plt = plot(sol_sig.t, RNA_sig, lw=3, label="Vsig (avg=$(round(avg_sig,digits=2)))")
plot!(plt, sol_min.t, RNA_min, lw=2, linestyle=:dash, label="Vmin (avg=$(round(avg_min,digits=2)))")
plot!(plt, sol_max.t, RNA_max, lw=2, linestyle=:dot, label="Vmax (avg=$(round(avg_max,digits=2)))")
xlabel!("Time")
ylabel!("[RNA]")
title!("RNA production: Vsig vs Vmin vs Vmax (example params)")
display(plt)

# ===========================
# Plot Vsig(t) for optimized control
# ===========================
function Vsig(t, A, ω, ϕ)
    Vmean = (Vmax + Vmin) / 2
    raw = Vmean + A * sin(ω*t + ϕ)
    return clamp(raw, Vmin, Vmax)
end

# use your optimized sinusoidal parameters
Aopt, wopt, phopt = A_opt, ω_opt, ϕ_opt   # already computed above

# time grid
tplot = LinRange(tspan[1], tspan[2], 1000)

# compute Vsignal
Vvals = Vsig.(tplot, Aopt, wopt, phopt)

# plot
pltV = plot(tplot, Vvals, lw=3, label="Vsig(t)",
            xlabel="Time", ylabel="V(t)",
            title="Optimized Sinusoidal Control Vsig(t)")
display(pltV)


# ===========================
# Plot ALL species in the Vmin simulation
# ===========================

species_names = ["α", "Cp", "αCp", "CppC", "RNA"]

plt_all = plot(title="All Species Under Vmin",
               xlabel="Time",
               ylabel="Concentration",
               lw=2)

for idx in 1:5
    species_vals = [sol_min.u[i][idx] for i in 1:length(sol_min.t)]
    plot!(plt_all, sol_min.t, species_vals, label=species_names[idx])
end

display(plt_all)




using BlackBoxOptim

function find_optimal_sinusoidal(params; verbose=false)

    # Objective to maximize
    function obj(x)
        A, ω, ϕ = x
        sol = simulate_sinusoidal(params, n0, tspan, A, ω, ϕ;
                                  saveat=SAVEAT_FAST,
                                  abstol=1e-6, reltol=1e-6)
        if sol.retcode != :Success
            return -1e9
        end
        return avg_RNA_from_sol(sol)
    end

    # BlackBoxOptim wants a vector of (low, high) tuples
    search_range = [
        (0.0, (Vmax - Vmin)/2),   # amplitude
        (0.005, 1.0),             # frequency
        (0.0, 2π)                 # phase
    ]

    ############################################
    # GLOBAL search with differential evolution
    ############################################

    res_global = bboptimize(
        (x -> -obj(x));                 # minimize → maximize obj
        Method = :adaptive_de_rand_1_bin_radiuslimited,
        SearchRange = search_range,     # FIXED FORMAT
        NumDimensions = 3,
        MaxSteps = 900,
        TraceMode = :silent
    )

    x_best = best_candidate(res_global)

    ############################################
    # LOCAL refinement (optional)
    ############################################

    obj_local(x) = -obj(x)

    lower = [r[1] for r in search_range]
    upper = [r[2] for r in search_range]

    res_local = optimize(
        obj_local, lower, upper, x_best,
        Fminbox(NelderMead()),
        Optim.Options(iterations=600, show_trace=verbose)
    )

    A_opt, ω_opt, ϕ_opt = res_local.minimizer

    ############################################
    # FINAL evaluation
    ############################################

    sol_final = simulate_sinusoidal(
        params, n0, tspan, A_opt, ω_opt, ϕ_opt;
        saveat=SAVEAT,
        abstol=1e-8, reltol=1e-8
    )

    avg = avg_RNA_from_sol(sol_final)

    return (A_opt, ω_opt, ϕ_opt), avg
end










# ===========================
# Plot ALL species in the Vsig simulation    
# Names of the 5 tracked species
species_names = ["α", "Cp", "αCp", "CppC", "RNA"]

# Create the plot object
plt_sig = plot(
    title = "All Species Under Vsigmoid",
    xlabel = "Time",
    ylabel = "Concentration",
    lw = 2
)

# Loop through species 1 to 5 and extract values
for idx in 1:5
    species_vals = [sol_sig.u[i][idx] for i in 1:length(sol_sig.t)]
    plot!(plt_sig, sol_sig.t, species_vals, label = species_names[idx])
end

# Display the combined figure
display(plt_sig)






# ===========================
# Plot ALL species in the Vsig simulation    
# Names of the 5 tracked species
species_names = ["α", "Cp", "αCp", "CppC", "RNA"]

# Create the plot object
plt_sig = plot(
    title = "All Species Under Vsigmoid",
    xlabel = "Time",
    ylabel = "Concentration",
    lw = 2
)

# Loop through species 1 to 5 and extract values
for idx in 1:5
    species_vals = [sol_sig.u[i][idx] for i in 1:length(sol_sig.t)]
    plot!(plt_sig, sol_sig.t, species_vals, label = species_names[idx])
end

# Display the combined figure
display(plt_sig)

############################
# THREE-PANEL RNA SUMMARY  #
############################

# --- Time grid for plotting ---
tplot = sol_sig.t
tplot = LinRange(tspan[1], tspan[2], 1000)
# --- Optimal volume signal ---
Vopt_vals = Vsig.(tplot, A_opt, ω_opt, ϕ_opt)

# --- Species under Vopt ---
tplot = LinRange(tspan[1], tspan[2], 1000)

α_opt    = [sol_sig(t)[1] for t in tplot]
Cp_opt   = [sol_sig(t)[2] for t in tplot]
αCp_opt  = [sol_sig(t)[3] for t in tplot]
CppC_opt = [sol_sig(t)[4] for t in tplot]
RNA_opt  = [sol_sig(t)[5] for t in tplot]

# --- RNA under constant volumes ---
RNA_Vmin  = [sol_min(t)[5]  for t in tplot]
RNA_Vmax  = [sol_max(t)[5]  for t in tplot]
RNA_Vmean = [sol_mean(t)[5] for t in tplot]

# Vmean simulation
Vmean = (Vmin + Vmax) / 2
sol_mean = simulate_constant(params_example, n0, tspan, Vmean;
                             saveat=SAVEAT, abstol=1e-8, reltol=1e-8)
RNA_Vmean = [sol_mean(t)[5] for t in tplot]

# --- RNA averages ---
avg_RNA_Vopt  = avg_RNA_from_sol(sol_sig)
avg_RNA_Vmin  = avg_RNA_from_sol(sol_min)
avg_RNA_Vmax  = avg_RNA_from_sol(sol_max)
avg_RNA_Vmean = avg_RNA_from_sol(sol_mean)

# ==========================
# Build the 3-panel layout
# ==========================
plt = plot(layout = @layout([a; b; c]), size=(900, 900))

# -------- Top panel: Vopt(t)
plot!(
    plt[1],
    tplot, Vopt_vals,
    lw = 3,
    label = "Vopt(t)"
)
ylabel!(plt[1], "Volume V(t)")
title!(plt[1], "Optimal Volume Control")
ylims1 = 110
annotate!(plt[1],
    (first(tplot),
     ylims1 - 0.02*(ylims1 - 0),
     text("(a)", 18, :black, :left))
)

# -------- Middle panel: species under Vopt
plot!(plt[2], tplot, α_opt,    lw=2, label="α")
plot!(plt[2], tplot, Cp_opt,   lw=2, label="Cp")
plot!(plt[2], tplot, αCp_opt,  lw=2, label="αCp")
plot!(plt[2], tplot, CppC_opt, lw=2, label="CppC")
plot!(plt[2], tplot, RNA_opt,  lw=3, label="RNA")
ylims2 = 280
annotate!(plt[2],
    (first(tplot),
     ylims2 - 0.02*(ylims2 - 0),
     text("(b)", 18, :black, :left))
)
ylabel!(plt[2], "Number of Molecules")
title!(plt[2], "Species Dynamics under Optimal Control")

# -------- Bottom panel: RNA comparison (averages in legend)
plot!(plt[3], tplot, RNA_opt,
      lw=3,
      label = "RNA – Vopt (avg = $(round(avg_RNA_Vopt, digits=2)))")

plot!(plt[3], tplot, RNA_Vmin,
      lw=2, ls=:dot,
      label = "RNA – Vmin (avg = $(round(avg_RNA_Vmin, digits=2)))")

plot!(plt[3], tplot, RNA_Vmax,
      lw=2, ls=:dash,
      label = "RNA – Vmax (avg = $(round(avg_RNA_Vmax, digits=2)))")

plot!(plt[3], tplot, RNA_Vmean,
      lw=2, ls=:dashdot,
      label = "RNA – Vmean (avg = $(round(avg_RNA_Vmean, digits=2)))")

xlabel!(plt[3], "Time")
ylabel!(plt[3], "RNA Molecules")
title!(plt[3], "RNA(t) Comparison Across Controls")
ylims3 = 250
annotate!(plt[3],
    (first(tplot),
     ylims3 - 0.02*(ylims3 - 0),
     text("(c)", 18, :black, :left))
)
display(plt)

# --------------------------
# Save to PNG
# --------------------------
savefig(plt, "RNA_model_three_panel_summary.png")



RNA_Vmin  = [sol_min(t)[5]  for t in tplot]
RNA_Vmax  = [sol_max(t)[5]  for t in tplot]
RNA_Vmean = [sol_mean(t)[5] for t in tplot]