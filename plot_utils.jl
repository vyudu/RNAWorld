# Plot all C curves together
"""
RNA concentration trajectory under each volume condition.
"""
function target_species_plot(sols; idx = 5, Vmin, Vmax, species_name = "RNA")
    (sol_opt, sol_max, sol_min) = sols
    pC = plot(sol_opt, idxs = idx, label="$species_name (optimal control)", lw=2)
    plot!(sol_max, idxs = idx, label="$species_name (V = $Vmax)", lw=2, ls=:dot)
    plot!(sol_min, idxs = idx, label="$species_name (V = $Vmin)", lw=2, ls=:dot)
    # plot!(sol_mean, idxs = idxs, label="Constant Vmean", lw=2, ls=:dashdot)
    xlabel!("Time")
    ylabel!("Concentration $species_name")
    title!("$species_name(t) under different controls")
end

"""
Concentration trajectory for all species under a given volume condition.
"""
function all_species_plot(sol; labels = ["α" "Cp" "αCp" "CppC" "RNA"])
    plot(sol, label = labels)
    xlabel!("Time")
    ylabel!("Concentration")
    title!("Species concentrations under optimal volume control")
end

"""
Plot optimal V(t). Takes in: vector of time points, vector of volumes.
"""
function volume_plot(V, ts; Vmin, Vmax)
    pV = plot(ts, V.(ts), label="Optimal V(t)", lw=2)
    xlabel!("Time")
    ylabel!("Volume V(t)")

    hline!([Vmin], linestyle=:dash, color=:red, label="Vmin")
    hline!([Vmax], linestyle=:dash, color=:green, label="Vmax")

    title!("Optimal Control V(t)")
end

function simulation_plot(sols, Vs; idx = 5, Vmin = 0.1, Vmax = 10., species_name = "RNA")
    (; opt, min, max) = sols
    pR = target_species_plot(sols; idx, Vmin, Vmax, species_name)
    pC = all_species_plot(opt)
    pV = volume_plot(Vs, opt.t; Vmin, Vmax)
    plot(pR, pC, pV, layout=(3,1), size=(600,1200))
end
