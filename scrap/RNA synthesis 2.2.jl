using Catalyst
using DifferentialEquations
using Plots
using Random

@enum Nucleotides A U C G

# Simple RNA generator
function generate_random_rna(size::Int)
    rna_sequence = [rand(instances(Nucleotides)) for _ in 1:size]
end

# Additional parameters
K_m = 1.06   # mM
v_max = 19.5 # h^-1
Ki = 24.7    # mM
p = [K_m, v_max, Ki]

function K_eff(X, αX, V)
    K_m * (1 + (X + αX)/(V*K_i))    
end

size = 100  # Change this value to generate a different sized RNA sequence
random_rna = generate_random_rna(size)
println("Random RNA sequence of size $size: $random_rna")

# Define the species
@variables t
@species begin 
    RNA(t) 
    α(t)
    αCp(t) 
    αGp(t) 
    αUp(t) 
    αAp(t) 

    Cp(t) 
    Gp(t) 
    Up(t) 
    Ap(t) 

    CppC(t) 
    CppG(t) 
    CppU(t) 
    CppA(t) 
    GppG(t) 
    GppA(t) 
    GppU(t)
    UppU(t) 
    UppA(t) 
    AppA(t) 
end
@parameters K_m v_max Ki

# Function to define V_func(t) for a given period
function V_func(t, period)
    return 5.05 + 4.95 * sin(2 * pi * t / period - 0.96)
end

# Define the reactions
rxns = @reaction_network begin
    @parameters begin
        K_mC K_mG K_mA K_mU
        v_max Ki
        τ 
    end
    @variables begin
        V(t, τ) ~ 5.05 + 4.95 * sin(2π*t / τ)
    end
    k2/V^2, 2αCp --> CppC + α                        # Formation of the catalyst (dimer of Cp)
    k3/V^2, Cp + α --> αCp                           # Formation of Activated Cystine
    k4/V, CppC --> αCp + Cp                        # Degradation of the Catalyzed dimer
    k5/V, αCp --> Cp + α                           # Degradation of the Activated Cystine
    k6/V, CppC + α --> 2αCp                        # Degradation of the catalyzed dimer with 2-aminoimidazole
    k7/V, 2αGp --> GppG + α                        # Formation of the catalyst (dimer of Gp)
    k8/V, Gp + α --> αGp                           # Formation of Activated guanosine
    k9/V, GppG --> αGp + Gp                        # Degradation of the Catalyzed dimer
    k10/V, αGp --> Gp + α                           # Degradation of the Activated guanosine
    k11/V, GppG + α --> 2αGp                        # Degradation of the catalyzed dimer with 2-aminoimidazole
    k12/V, 2αUp --> UppU + α                         # Formation of the catalyst (dimer of Up)
    k13/V, Up + α --> αUp                           # Formation of Activated uridine 
    k14/V, UppU --> αUp + Up                        # Degradation of the Catalyzed dimer
    k15/V, αUp --> Up + α                           # Degradation of the Activated uridine 
    k16/V, UppU + α --> 2αUp                        # Degradation of the catalyzed dimer with 2-aminoimidazole
    k17/V^2, 2αAp --> AppA + α                        # Formation of the catalyst (dimer of Ap)
    k18/V^2, Ap + α --> αAp                           # Formation of Activated Adenosine
    k19/V, AppA --> αAp + Cp                        # Degradation of the Catalyzed dimer
    k20/V, αAp --> Ap + α                           # Degradation of the Activated Adenosine
    k21/V^2, AppA + α --> 2αAp                        # Degradation of the catalyzed dimer with 2-aminoimidazole
    k22/V, CppG --> αGp + Cp                        # Degradation of CppG into AGp and Cp
    k23/V, CppG --> αCp + Gp                        # Degradation of CppG into αCp and Gp
    k24/V, CppU --> αUp + Cp                        # Degradation of CppU into AUp and Cp
    k25/V, CppU --> αCp + Up                        # Degradation of CppU into αCp and Up
    k26/V, CppA --> αAp + Cp                        # Degradation of CppA into AAp and Cp
    k27/V, CppA --> αCp + Ap                        # Degradation of CppA into αCp and Ap
    k28/V, GppU --> αGp + Up                        # Degradation of GppU into AGp and Up
    k29/V, GppU --> αUp + Gp                        # Degradation of GppU into AUp and Gp
    k30/V, GppA --> αGp + Ap                        # Degradation of GppA into AGp and Ap
    k31/V, GppA --> αAp + Gp                        # Degradation of GppA into AAp and Gp
    k32/V, UppA --> αAp + Up                        # Degradation of UppA into AAp and Up
    k33/V, UppA --> αUp + Ap                        # Degradation of UppA into AUp and Ap  
    k34/V^2, αCp + αGp --> CppG + α                    # Formation of CppG from αCp and AGp
    k35/V^2, αCp + αUp --> CppU + α                   # Formation of CppU from αCp and AUp
    k36/V^2, αCp + αAp --> CppA + α                   # Formation of CppA from αCp and AAp
    k37/V^2, αGp + αUp --> GppU + α                   # Formation of GppU from AGp and AUp
    k38/V^2, αGp + αAp --> GppA + α                   # Formation of GppA from AGp and AAp
    k39/V^2, αUp + αAp --> UppA + α                   # Formation of UppA from AUp and AAp

    mm(CppC, v_max, Keff(Cp, αCp, V)), CppC --> Cp + RNA     # Catalyzed polymer growth
    mm(GppG, v_max, Keff(Gp, αGp, V)), GppG --> Gp + RNA     # Catalyzed polymer growth
    mm(UppU, v_max, Keff(Up, αUp, V)), UppU --> Up + RNA     # Catalyzed polymer growth
    mm(AppA, v_max, Keff(Ap, αAp, V)), AppA --> Ap + RNA     # Catalyzed polymer growth 
    mm(CppG, v_max, Keff(Cp, αGp, V)), CppG --> Gp + RNA # Addition of Cp when template is 3' GC 5'
    mm(CppG, v_max, Keff(Gp, αCp, V)), CppG --> Cp + RNA # Addition of Gp when template is 3' CG 5'
    mm(CppU, v_max, Keff(Cp, αUp, V)), CppU --> Up + RNA # Addition of Cp when template is 3' GA 5'
    mm(CppU, v_max, Keff(Up, αCp, V)), CppU --> Cp + RNA # Addition of Ap when template is 3' AG 5'
    mm(CppA, v_max, Keff(Cp, αAp, V)), CppA --> Ap + RNA # Addition of Gp when template is 3' GU 5'
    mm(CppA, v_max, Keff(Ap, αCp, V)), CppA --> Cp + RNA # Addition of Gp when template is 3' UG 5'
    mm(GppA, v_max, Keff(Gp, αAp, V)), GppA --> Ap + RNA # Addition of Gp when template is 3' CU 5'
    mm(GppA, v_max, Keff(Ap, αGp, V)), GppA --> Gp + RNA # Addition of Gp when template is 3' UC 5'
    mm(GppU, v_max, Keff(Gp, αUp, V)), GppU --> Up + RNA # Addition of Gp when template is 3' CA 5'
    mm(GppU, v_max, Keff(Up, αGp, V)), GppU --> Gp + RNA # Addition of Gp when template is 3' AC 5'
    mm(UppA, v_max, Keff(Up, αAp, V)), UppA --> Ap + RNA # Addition of Gp when template is 3' AU 5'
    mm(UppA, v_max, Keff(Ap, αUp, V)), UppA --> Up + RNA # Addition of Gp when template is 3' UA 5'
end

# Initial conditions for the species
# Initial conditions for the species
u0 = [
    10.0,  # RNA(t) 
    0.0,   # α(t)
    0.0,   # αCp(t) 
    0.0,   # αGp(t) 
    1.0,   # αUp(t) 
    10.0,  # αAp(t) 
    0.0,   # Cp(t)       
    0.0,   # Gp(t)
    0.0,   # Up(t)
    0.0,   # Ap(t)
    10.0,  # CppC(t)
    10.0,  # CppG(t)       
    0.0,   # CppU(t)
    0.0,   # CppA(t)
    0.0,   # GppG(t)
    0.0,   # GppA(t)
    0.0,   # GppU(t)
    0.0,   # UppU(t)
    0.0,   # UppA(t)
    0.0    # AppA(t)
]             
              
rate_constants = begin
    k1 = 4.55e-3
    k3 = 0.1
    k4 = 0.167
    k5 = 3.33e-3
    k6 = 0.183 
    k7 = 4.55e-3
    k8 = 0.1
    k9 = 0.167
    k10 = 3.33e-3
    k11 = 0.183
    k12 = 4.55e-3
    k13 = 0.1
    k14 = 0.167
    k15 = 3.33e-3
    k16 = 0.183
    k17 = 4.55e-3
    k18 = 0.1
    k19 = 0.167
    k20 = 3.33e-3
    k21 = 0.183
    k22 = 0.183
    k23 = 0.183
    k24 = 0.183
    k25 = 0.183
    k26 = 0.183
    k27 = 0.183
    k28 = 0.183
    k29 = 0.183
    k30 = 0.183
    k31 = 0.183
    k32 = 0.183
    k33 = 0.183
    k34 = 4.55e-3
    k35 = 4.55e-3
    k36 = 4.55e-3
    k37 = 4.55e-3
    k38 = 4.55e-3
    k39 = 4.55e-3
end


tspan = (0.0, 600.0)

function ode_func(dy, y, p, t)
    ACp, CppC, A, RNA, Cp, AGp, Gp, GppG, UppU, Up, AUp, AAp, Ap, AppA, CppG, CppU, CppA, UppA, GppA, GppU = y
    K_m, v_max, Ki = p

    # Get the current volume
    V = V_func(t)
    Keff1 = K_m * (1.0 + (Cp / V + ACp / V) / Ki)  # Effective Michaelis constant for CppC
    Keff2 = K_m * (1.0 + (Gp / V + AGp / V) / Ki)  # Effective Michaelis constant for GppG
    Keff3 = K_m * (1.0 + (Up / V + AUp / V) / Ki)  # Effective Michaelis constant for UppU
    Keff4 = K_m * (1.0 + (Ap / V + AAp / V) / Ki)  # Effective Michaelis constant for AppA
    Keff5 = K_m * (1.0 + (Cp / V + AGp / V) / Ki)  # Effective Michaelis constant for CppG
    Keff6 = K_m * (1.0 + (Gp / V + ACp / V) / Ki)  # Effective Michaelis constant for CppG (2nd reaction)
    Keff7 = K_m * (1.0 + (Cp / V + AUp / V) / Ki)  # Effective Michaelis constant for CppU
    Keff8 = K_m * (1.0 + (Up / V + ACp / V) / Ki)  # Effective Michaelis constant for CppU (2nd reaction)
    Keff9 = K_m * (1.0 + (Cp / V + AAp / V) / Ki)  # Effective Michaelis constant for CppA
    Keff10 = K_m * (1.0 + (Ap / V + ACp / V) / Ki) # Effective Michaelis constant for CppA (2nd reaction)
    Keff11 = K_m * (1.0 + (Gp / V + AAp / V) / Ki) # Effective Michaelis constant for GppA
    Keff12 = K_m * (1.0 + (Ap / V + AGp / V) / Ki) # Effective Michaelis constant for GppA (2nd reaction)
    Keff13 = K_m * (1.0 + (Gp / V + AUp / V) / Ki) # Effective Michaelis constant for GppU
    Keff14 = K_m * (1.0 + (Up / V + AGp / V) / Ki) # Effective Michaelis constant for GppU (2nd reaction)
    Keff15 = K_m * (1.0 + (Ap / V + AUp / V) / Ki) # Effective Michaelis constant for UppA
    Keff16 = K_m * (1.0 + (Up / V + AAp / V) / Ki) # Effective Michaelis constant for UppA (2nd reaction)

    # Define the rate constants as functions of time

    # Define the reaction rates
    r1 = k1 * (ACp / V)^2                            # 2ACp --> CppC + A
    r3 = k3 * (Cp / V) * (A / V)                     # k3, Cp + α --> αCp                        
    r4 = k4 * (CppC / V)                             # k4, CppC --> αCp + Cp                     
    r5 = k5 * (ACp / V)                              # k5, αCp --> Cp + α                        
    r6 = k6 * (CppC / V) * (A / V)                   # k6, CppC + α --> 2αCp                     
    r7 = k7 * (AGp / V)^2                            # k7, 2αGp --> GppG + α 
    r8 = k8 * (Gp / V) * (A / V)                     # k8, Gp + α --> αGp                        
    r9 = k9 * (GppG / V)                            # k9, GppG --> αGp + Gp                     
    r10 = k10 * (AGp / V)                            # k10, αGp --> Gp + α                       
    r11 = k11 * (GppG / V) * (A / V)                 # k11, GppG + α --> 2αGp                    
    r12 = k12 * (AUp / V)^2                          # k12, 2αUp --> UppU + α                    
    r13 = k13 * (Up / V) * (A / V)                   # k13, Up + α --> αUp                       
    r14 = k14 * (UppU / V)                           # k14, UppU --> αUp + Up                    
    r15 = k15 * (AUp / V)                            # k15, αUp --> Up + α                       
    r16 = k16 * (UppU / V) * (A / V)                 # k16, UppU + α --> 2αUp                    
    r17 = k17 * (AAp / V)^2                          # k17, 2αAp --> AppA + α                    
    r18 = k18 * (Ap / V) * (A / V)                   # k18, Ap + α --> αAp 
    r19 = k19 * (AppA / V)                           # k19, AppA --> αAp + Cp                    
    r20 = k20 * (AAp / V)                            # k20, αAp --> Ap + α                       
    r21 = k21 * (AppA / V) * (A / V)                 # k21, AppA + α --> 2αAp                    
    r22 = k22 * (CppG / V)                           # k22, CppG --> αGp + Cp                    
    r23 = k23 * (CppG / V)                           # k23, CppG --> αCp + Gp                    
    r24 = k24 * (CppU / V)                           # k24, CppU --> αUp + Cp                    
    r25 = k25 * (CppU / V)                           # k25, CppU --> αCp + Up 
    r26 = k26 * (CppA / V)                           # k26, CppA --> αAp + Cp                    
    r27 = k27 * (CppA / V)                           # k27, CppA --> αCp + Ap                    
    r28 = k28 * (GppU / V)                           # k28, GppU --> αGp + Up                    
    r29 = k29 * (GppU / V)                           # k29, GppU --> αUp + Gp                    
    r30 = k30 * (GppA / V)                           # k30, GppA --> αGp + Ap                    
    r31 = k31 * (GppA / V)                           # k31, GppA --> αAp + Gp                    
    r32 = k32 * (UppA / V)                           # k32, UppA --> αAp + Up 
    r33 = k33 * (UppA / V)                           # k33, UppA --> αUp + Ap                    
    r34 = k34 * (ACp / V) * (AGp / V)                # k34, αCp + αGp --> CppG + α               
    r35 = k35 * (ACp / V) * (AUp / V)                # k35, αCp + αUp --> CppU + α               
    r36 = k36 * (ACp / V) * (AAp / V)                # k36, αCp + αAp --> CppA + α               
    r37 = k37 * (AGp / V) * (AUp / V)                # k37, αGp + αUp --> GppU + α               
    r38 = k38 * (AGp / V) * (AAp / V)                # k38, αGp + αAp --> GppA + α               
    r39 = k39 * (AUp / V) * (AAp / V)                # k39, αUp + αAp --> UppA + α               
    R1 = v_max * (CppC / V) / (Keff1 + (CppC / V))   # mm(CppC, v_max, Keff1), CppC --> Cp + RNA 
    R2 = v_max * (GppG / V) / (Keff2 + (GppG / V))   # mm(GppG, v_max, Keff2), GppG --> Gp + RNA 
    R3 = v_max * (UppU / V) / (Keff3 + (UppU / V))   # mm(UppU, v_max, Keff3), UppU --> Up + RNA 
    R4 = v_max * (AppA / V) / (Keff4 + (AppA / V))   # mm(AppA, v_max, Keff4), AppA --> Ap + RNA 
    R5 = v_max * (CppG / V) / (Keff5 + (CppG / V))   # mm(CppG, v_max, Keff5), CppG --> Gp + RNA 
    R6 = v_max * (CppG / V) / (Keff6 + (CppG / V))   # mm(CppG, v_max, Keff6), CppG --> Cp + RNA 
    R7 = v_max * (CppU / V) / (Keff7 + (CppU / V))   # mm(CppU, v_max, Keff7), CppU --> Up + RNA 
    R8 = v_max * (CppU / V) / (Keff8 + (CppU / V))   # mm(CppU, v_max, Keff8), CppU --> Cp + RNA 
    R9 = v_max * (CppA / V) / (Keff9 + (CppA / V))   # mm(CppA, v_max, Keff9), CppA --> Ap + RNA  
    R10 = v_max * (CppA / V) / (Keff10 + (CppA / V)) # mm(CppA, v_max, Keff10), CppA --> Cp + RNA
    R11 = v_max * (GppA / V) / (Keff11 + (GppA / V)) # mm(GppA, v_max, Keff11), GppA --> Ap + RNA
    R12 = v_max * (GppA / V) / (Keff12 + (GppA / V)) # mm(GppA, v_max, Keff12), GppA --> Gp + RNA 
    R13 = v_max * (GppU / V) / (Keff13 + (GppU / V)) # mm(GppU, v_max, Keff13), GppU --> Up + RNA
    R14 = v_max * (GppU / V) / (Keff14 + (GppU / V)) # mm(GppU, v_max, Keff14), GppU --> Gp + RNA
    R15 = v_max * (UppA / V) / (Keff15 + (UppA / V)) # mm(UppA, v_max, Keff15), UppA --> Ap + RNA 
    R16 = v_max * (UppA / V) / (Keff16 + (UppA / V)) # mm(UppA, v_max, Keff16), UppA --> Up + RNA 

    # r2 r8 r14 r20 
    # Define the ODEs for concentrations             
    dy[1] = -2r1 + r3 - r5 + 2r6 - r34 - r35 - r36               # dACp/dt
    dy[2] = r1 - R1 - r4 - r6                  # dCppC/dt
    dy[3] = r1 - r3 + r5 - r6 - r9             # dA/dt
    dy[4] = R1 + R2 + R3 + R4 + R5 + R6 + R7 + R8 + R9 + R10 + R11 + R12 + R13 + R14 + R15 + R16  # dRNA/dt6
    dy[5] = R1 + r4 - r3 + r5                  # dCp/dt
    dy[6] = r22 - r34 - r37 - r38              # dAGp/dt
    dy[7] = r23 + r29 + r31 - r6 - r37 - r38   # dGp/dt
    dy[8] = r34 - r28 - r29                    # dGppG/dt
    dy[9] = r28 - r32 - r33                    # dUppU/dt
    dy[10] = r24 + r30 - r32 - r33             # dUp/dt
    dy[11] = r24 - r35 - r37 - r39             # dAUp/dt
    dy[12] = r26 + r31 - r36 - r38 - r39       # dAAp/dt
    dy[13] = r25 + r27 - r36 - r39             # dAp/dt
    dy[14] = -r19 - r21 + r39                  # dAppA/dt
    dy[15] = r37 + r38 + r25 + r26 - R5 - R6   # dCppG/dt
    dy[16] = r38 + r39 + r27 + r28 - R7 - R8   # dCppU/dt
    dy[17] = r39 + r34 + r29 + r30 - R9 - R10  # dCppA/dt
    dy[18] = r38 + r39 + r35 + r32 - R15 - R16 # dUppA/dt
    dy[19] = r37 + r39 + r33 + r34 - R11 - R12 # dGppA/dt
    dy[20] = r37 + r38 + r31 + r32 - R13 - R14 # dGppU/dt
end

# Arrays to store results
periods = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 50, 60, 80, 100, 120, 150, 200, 240, 300, 600, 1200]
rna_results = []

# Solve the ODE for each period and store the results
for period in periods
    V_func_period = t -> V_func(t, period)
    prob = ODEProblem((dy, y, p, t) -> ode_func(dy, y, p, t, V_func_period), u0, tspan, p)
    sol = solve(prob, Vern8())
    push!(rna_results, maximum(sol[4, :]))  # Store the maximum RNA produced
end

# Identify the period with the highest RNA production
max_rna = maximum(rna_results)
optimal_period = periods[argmax(rna_results)]

# Plot the results
scatter(periods, rna_results, xlabel="Period", ylabel="Maximum RNA Produced", legend=false)
title!("Maximum RNA Produced vs Period for ACp 20 mM")

# Display the optimal period
println("The period that achieves the highest RNA production is: $optimal_period")
println("The highest RNA production achieved is: $max_rna")

max_rna = maximum(sol[4, :])
time_to_max_rna = sol.t[argmax(sol[4, :])]



# Select a single period
period = 50  # You can change this to any desired period
V_func_period = t -> V_func(t, period)

# Solve the ODE
prob = ODEProblem((dy, y, p, t) -> ode_func(dy, y, p, t, V_func_period), u0, tspan, p)
sol = solve(prob, Vern8())

# Plot the results
plot(sol, vars=(1:20), xlabel="Time", ylabel="Concentration", label=["ACp" "CppC" "A" "RNA" "Cp" "AGp" "Gp" "GppG" "UppU" "Up" "AUp" "AAp" "Ap" "AppA" "CppG" "CppU" "CppA" "UppA" "GppA" "GppU"])
title!("Concentration of Species Over Time for Period 30")
