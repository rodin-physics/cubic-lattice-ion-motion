include("../src/main.jl")

dirs = ["Data", "Data/1D_Loss"]
[isdir(d) ? nothing : mkdir(d) for d in dirs]
## PARAMETERS
μ = [Inf, Inf, 1]       # Mass of the particle in the units of chain masses
α = 2                   # Lattice constant
ω_perp = sqrt(99 / 4)   # Frequency of the chain spring for transverse mass displacements
ω_par = sqrt(99 / 4)    # Frequency of the chain spring for longitudinal mass displacements

size_x = 1
size_y = 1
size_z = 200

# Interaction
Φ0 = 0.01                   # Amplitude of the Gaussian interaction
s_vals = [0, 1]             # Impact parameters
λ_vals = [1 / 2, 1, 2, 4]   # Extent of the Gaussian interaction
ν_vals = [0, 1]             # Frequency of the confining potential
params = [(s, λ, ν) for s in s_vals, λ in λ_vals, ν in ν_vals] |> vec

# Simulation settings
nPts = 50                   # Number of points
δτ = 1e-4                   # Time step

println("Starting calculations...")
## SIMULATIONS
Threads.@threads for p in params
    s = p[1]
    λ = p[2]
    ν = p[3]

    if λ == 1 / 2 || λ == 1
        speed_min = 2
        speed_max = 80
    elseif λ == 2
        speed_min = 4
        speed_max = 80
    else
        speed_min = 8
        speed_max = 80
    end

    init_speeds = range(speed_min, speed_max, length = nPts)
    dense_speeds = range(speed_min, speed_max, length = nPts * 10)

    # Interaction function
    @inline function Φ(r)
        res = Φ0 * exp(-dot(r, r) / 2 / λ^2)
        return res
    end

    if !isfile("Data/1D_Loss/1D_Gaussian_Loss_s$(s)_λ$(λ)_ν$(ν).jld2")

        self_coupling = [
            Coupling((0, 0, 0), 1, 1, ν^2 + 2 * ω_perp^2),
            Coupling((0, 0, 0), 2, 2, ν^2 + 2 * ω_perp^2),
            Coupling((0, 0, 0), 3, 3, ν^2 + 2 * ω_par^2),
        ]

        neighbor_coupling = [
            Coupling((0, 0, 1), 1, 1, -ω_perp^2),
            Coupling((0, 0, -1), 1, 1, -ω_perp^2),
            Coupling((0, 0, 1), 2, 2, -ω_perp^2),
            Coupling((0, 0, -1), 2, 2, -ω_perp^2),
            Coupling((0, 0, 1), 3, 3, -ω_par^2),
            Coupling((0, 0, -1), 3, 3, -ω_par^2),
        ]

        coupling = vcat(self_coupling, neighbor_coupling)

        # Dynamical matrix
        DynamicalMatrix = dynamical_matrix(coupling)

        sys = system(size_x, size_y, size_z, coupling)
        atoms = [(1, 1, 1)]           # The moving particle interacts with a single atom

        loss = zeros(nPts)
        numerical_loss = zeros(nPts)
        @showprogress for ii in eachindex(loss)
            init_pos = -7 * λ
            init_speed = init_speeds[ii]
            σ = [0, s, init_pos]
            σ_dot = [0, 0, init_speed]

            ρ_init = zeros(length(sys[1]))
            ρ_dot_init = zeros(length(sys[1]))

            current_state = (ρ_init, ρ_dot_init, σ, σ_dot)
            while current_state[3][3] < 7 * λ
                current_state = RKstep(μ, α, sys, atoms, Φ, current_state, δτ)
            end
            final_speed = current_state[4][3]
            loss[ii] = (init_speed^2) / (8 * π^2) - (final_speed^2) / (8 * π^2)

            numerical_loss[ii] = numerical_Δ(Φ0, λ, s, init_speed, DynamicalMatrix)[1]

        end
        analytic_loss = [
            sum(Δ_1D_analytic(speed, Φ0, λ, s, ω_par, ω_perp, ν)) for speed in dense_speeds
        ]

        save_object(
            "Data/1D_Loss/1D_Gaussian_Loss_s$(s)_λ$(λ)_ν$(ν).jld2",
            (s, λ, ν, init_speeds, loss, numerical_loss, dense_speeds, analytic_loss),
        )
    end
end
