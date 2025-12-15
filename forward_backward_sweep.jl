##########################################
### Solve Pontryagin Maximum Principle ###
##########################################
function forward_backward_sweep(model::ReactionModel, init_controller, solver, x0, λ0, tspan; tol = 1e-5, maxiters = 1000)
    # x_i, λ_i, u_i 
    # x_{i+1}, λ_{i+1}, u_{i+1}
    
    (; state_dynamics!, adjoint_dynamics!, ks) = model
    uᵢ = init_controller
    function state!(du, u, p, t)
        state_dynamics!(du, u, ks, uᵢ(t))
    end
    prob_fwd = ODEProblem(state!, x0, tspan)
    xᵢ = solve(prob_fwd, solver; dtmax = 1.)

    function costate!(dλ, λ, p, t)
        adjoint_dynamics!(dλ, xᵢ(t), λ, ks, uᵢ(t))
    end
    prob_bwd = ODEProblem(costate!, λ0, (tspan[2], tspan[1]))
    λᵢ = solve(prob_bwd, solver; dtmax = 1.)
    uᵢ = construct_controller(model, xᵢ, λᵢ)

    for sweep in 1:maxiters
        x_prev = xᵢ
        λ_prev = λᵢ
        u_prev = uᵢ

        xᵢ = solve(prob_fwd, solver; dtmax = 1.)
        λᵢ = solve(prob_bwd, solver; dtmax = 1.)
        uᵢ = construct_controller(model, xᵢ, λᵢ)

        is_below_tol(xᵢ, x_prev; tol) && is_below_tol(λᵢ, λ_prev; tol) && is_below_tol(uᵢ, u_prev; tol) && break
    end
    return uᵢ, xᵢ, λᵢ
end

function is_below_tol(sol, sol_prev; tol = 1e-4)
    tsteps = sol.t
    err = 0
    for (i, t) in enumerate(tsteps)
        err += norm(sol(t) - sol_prev(t))
    end
    err < tol
end

function construct_controller(model::ReactionModel, x, λ)
    u_vals = [optimal_V(model, x(t), λ(t)) for t in x.t]
    ConstantInterpolation(u_vals, x.t; extrapolation = ExtrapolationType.Extension)
end

function optimal_V(model::ReactionModel, n, λ)
    obj(V) = -Hamiltonian(model, n, λ, V)
    result = Optim.optimize(obj, 0.1, 10.0)
    return Optim.minimizer(result)
end

"""
Given a set of rate constants, a low and high volume, initial concentration, find the optimal control and plot the RNA trajectory for each condition (low V, high V, control V).
"""
function volume_simulation(model::ReactionModel, V_lo, V_hi, n0, λ0, tspan)
    (; state_dynamics!, ks) = model

    ode_func_hi!(dy, y, p, t) = state_dynamics!(dy, y, ks, V_hi)
    prob_hi = ODEProblem(ode_func_hi!, n0, tspan)
    sol_hi = solve(prob_hi, Rodas5P())

    ode_func_low!(dy, y, p, t) = state_dynamics!(dy, y, ks, V_lo)
    prob_low = ODEProblem(ode_func_low!, n0, tspan)
    sol_low = solve(prob_low, Rodas5P())

    controller, sol_opt, sol_co = forward_backward_sweep(model, Returns(1.), Vern7(), n0, λ0, tspan)

    controller, (; opt = sol_opt, min = sol_low, max = sol_hi)
end
