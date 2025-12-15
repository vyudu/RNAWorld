# ==========================================
# File: Optimize_Average_RNA_Model.jl
# Goal: Optimize control V(t) to maximize RNA Average (fast, stable version)
# ==========================================
using OrdinaryDiffEqTsit5, Optim, Trapz, Interpolations, Statistics, DataInterpolations
using Plots

# -------------------------
# Constants and Parameters
# -------------------------
const eps_div = 1e-9
Vmin, Vmax = 0.1, 100.0
tspan = (0.0, 100.0)
n0 = [50.0, 500.0, 50.0, 5.0, 0.0]          # [α, Cp, αCp, CppC, RNA]
params = (0.251188643150958, 1.0, 0.5011872336272722, 0.15848931924611134, 0.025118864315095794, 2.51188643150958, 0.7943282347242815, 0.025118864315095794)  # (k1,k2,k3,k4,k5,c6,Kd,D1)

# -------------------------
# Reaction Model
# -------------------------
function rates(n::AbstractVector{T}, V::T, p) where T
    α, Cp, αCp, CppC, RNA = n
    k1,k2,k3,k4,k5,c6,Kd,D1 = p

    r1 = k1 * (αCp^2) / (V^2)
    r2 = k2 * (α * CppC) / (V^2)
    r3 = k3 * CppC / V
    r4 = k4 * (α * Cp) / (V^2)
    r5 = k5 * αCp / V
    r6 = c6 * (1 + (Cp + αCp) / (Kd * V)) * (CppC / V)
    r7 = D1 * RNA  # RNA degradation
    return r1,r2,r3,r4,r5,r6,r7
end

function model!(du, n, p, t, V)
    r1,r2,r3,r4,r5,r6,r7 = rates(n, V, p)
    du[1] = r1 + r5 - r2 - r4           # α̇
    du[2] = r3 + r5 - r4 + r7           # Cṗ
    du[3] = 2r2 - 2r1 + r3 + r4 - r5 + r6   # αCṗ
    du[4] = r1 - r2 - r3 - r6           # CppĊ
    du[5] = r6 - r7                     # RNȦ
end

# -------------------------
# Sinusoidal control
# -------------------------
function sinusoidal_V(t, A, ω, ϕ)
    Vmean = (Vmax + Vmin) / 2
    raw = Vmean + A * sin(ω * t + ϕ)
    # Keep it within [Vmin, Vmax]
    return clamp(raw, Vmin, Vmax)
end

# Simulation for given sinusoidal parameters
function simulate_sinusoidal(params, n0, tspan, A, ω, ϕ)
    f!(du, u, p, t) = model!(du, u, p, t, sinusoidal_V(t, A, ω, ϕ))
    prob = ODEProblem(f!, n0, tspan, params)
    sol = solve(prob, Tsit5(); abstol=1e-6, reltol=1e-6, saveat=1.0)
    return sol
end

# -------------------------
# Optimization for A, ω, ϕ
# -------------------------
function find_optimal_sinusoidal(params)
    obj(x) = begin
        A, ω, ϕ = x
        sol = simulate_sinusoidal(params, n0, tspan, A, ω, ϕ)
        if sol.retcode != :Success
            return 1e6
        end
        RNA_vals = getindex.(sol.u, 5)
        avg_RNA = trapz(sol.t, RNA_vals) / (sol.t[end] - sol.t[1])
        return -avg_RNA
    end

    # Reasonable parameter bounds
    lower = [0.0, 0.01, 0.0]     # A, ω, ϕ lower bounds
    upper = [(Vmax - Vmin)/2, 1.0, 2π]  # upper bounds
    x0 = [10.0, 0.1, 0.0]

    result = optimize(obj, lower, upper, x0,
                      Fminbox(NelderMead()),
                      Optim.Options(iterations=800, show_trace=true))

    println("✅ Optimization complete.")
    println("Best parameters: A = $(result.minimizer[1]), ω = $(result.minimizer[2]), ϕ = $(result.minimizer[3])")
    println("Best mean RNA = ", -result.minimum)

    return result.minimizer
end

# Generate V(t) and RNA plots
Vsig = [sinusoidal_V(t, A_opt, ω_opt, ϕ_opt) for t in tvals]
plot(tvals, Vsig, lw=2, color=:blue, xlabel="Time", ylabel="V(t)", 
     title="Optimized Sinusoidal Control V(t)", label="V(t)")

# Simulate with optimized parameters
sol_sin = simulate_sinusoidal(params, n0, tspan, A_opt, ω_opt, ϕ_opt)
RNA_sin = getindex.(sol_sin.u, 5)
plot(sol_sin.t, RNA_sin, lw=2, xlabel="Time", ylabel="[RNA]", 
     title="RNA Production (Optimized Sinusoidal Control)", color=:blue)

# -------------------------
# Optimization Function
# -------------------------
function find_optimal_sinusoidal(params)
    obj(x) = begin
        A, ω, ϕ = x
        sol = simulate_sinusoidal(params, n0, tspan, A, ω, ϕ)
        if sol.retcode != :Success
            return 1e6
        end
        RNA_vals = getindex.(sol.u, 5)
        avg_RNA = trapz(sol.t, RNA_vals) / (sol.t[end] - sol.t[1])
        return -avg_RNA
    end

    lower = [0.0, 0.01, 0.0]
    upper = [(Vmax - Vmin)/2, 1.0, 2π]
    x0 = [10.0, 0.1, 0.0]

    result = optimize(obj, lower, upper, x0,
                      Fminbox(NelderMead()),
                      Optim.Options(iterations=800, show_trace=false))

    return result.minimizer, -result.minimum
end

function random_parameters(lbs, ubs)
    scale = rand(7)
    exponents = lbs + (ubs - lbs) .* scale
    return 10.^exponents
end

# --------------------------
# Simulate with constant volume
# --------------------------
function simulate_constant(params, n0, tspan, Vconst)
    f!(du, u, p, t) = model!(du, u, p, t, Vconst)
    prob = ODEProblem(f!, n0, tspan, params)
    sol = solve(prob, Tsit5(); abstol=1e-8, reltol=1e-8)
    return sol
end

function parameter_optimization(iters, n0, tspan)
    # -------------------------
    # Parameter Search (Exponential Scaling)
    # -------------------------
    lbs = [-1., -2., -2., -1., -2., -1., -1., -3.]
    ubs = [1., 0., 0., 1., 0., -0.7, 1., -0.3]

    for i in iters
        println("\n🧪 Trial $i / $n_trials")
        params_rand = random_parameters(lbs, ubs)

        Vparams, avg_RNA_sig = find_optimal_sinusoidal(params_rand)

        # Simulate constant Vmin and Vmax
        sol_min = simulate_constant(params_rand, n0, tspan, Vmin)
        sol_max = simulate_constant(params_rand, n0, tspan, Vmax)

        RNA_min = getindex.(sol_min.u, 5)
        RNA_max = getindex.(sol_max.u, 5)

        avg_RNA_min = trapz(sol_min.t, RNA_min) / (sol_min.t[end] - sol_min.t[1])
        avg_RNA_max = trapz(sol_max.t, RNA_max) / (sol_max.t[end] - sol_max.t[1])

        if (avg_RNA_sig > avg_RNA_min + 10) && (avg_RNA_sig > avg_RNA_max + 10)
            push!(successes, (
                params = params_rand,
                sinusoidal_avg = avg_RNA_sig,
                Vmin_avg = avg_RNA_min,
                Vmax_avg = avg_RNA_max,
                Vopt_params = Vparams
            ))
        end
    end

    successes
end

begin
    successes = parameter_optimization(10_000, n0, tspan)
    (ks, avg_opt, avg_min, avg_max, V_params) = first(successes)
end
