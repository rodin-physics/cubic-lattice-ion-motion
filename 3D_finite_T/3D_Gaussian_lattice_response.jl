include("../src/main.jl")
include("../src/plotting.jl")

## PARAMETERS
μ = [Inf, Inf, Inf]         # Mass of the particle in the units of framework masses
α = 5                       # Lattice constant
k1 = 15                     # Nearest neighbor force constant
k2 = 5                      # Next-nearest neighbor force constant

# System size
size_x = size_y = size_z = 40

# Interaction
λ = α / 4
Φ0 = 5000

@inline function Φ(r)
    res = Φ0 * exp(-dot(r, r) / 2 / λ^2)
    return res
end

# Initial and final positions of the particle
σ_init = [(size_x + 1) / 2 - 1, (size_y + 1) / 2 - 1, (size_z + 1) / 2 - 1] .* α
σ_final = [(size_x + 1) / 2 - 1, (size_y + 1) / 2 - 1, (size_z + 3) / 2 - 1] .* α

# Speed parameters
min_speed = 1                                   # Minimum speed
max_speed = 2 * (2 * π * √(k1 + 2 * k2) * α)    # Double the speed of sound
nSpeeds = 300
speeds = range(min_speed, max_speed, length = nSpeeds)

nPts = 600                                      # Number of points to be sampled during the trajectory
nStep = 150                                     # Number of steps used for calculating the initial relaxed config 

# LATTICE
# Nearest-neighbor and next-nearest neighbor displacements
N_disp = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
NN_disp = [
    (1, 1, 0),
    (-1, 1, 0),
    (1, -1, 0),
    (-1, -1, 0),
    (0, 1, 1),
    (0, -1, 1),
    (0, 1, -1),
    (0, -1, -1),
    (1, 0, 1),
    (-1, 0, 1),
    (1, 0, -1),
    (-1, 0, -1),
]

# Couplings for nearest and next-nearest atoms
N_couplings = vcat(
    [
        [Coupling(v, n, m, -k1 * v[n] * v[m] / dot(v, v)) for n = 1:3, m = 1:3] |> vec
        for v in N_disp
    ]...,
)
N_couplings = filter(x -> x.k != 0, N_couplings)

NN_couplings = vcat(
    [
        [Coupling(v, n, m, -k2 * v[n] * v[m] / dot(v, v)) for n = 1:3, m = 1:3] |> vec
        for v in NN_disp
    ]...,
)

NN_couplings = filter(x -> x.k != 0, NN_couplings)
# The zero coupling terms are filtered out

# Normalized vectors to nearest and next-nearest neighbors
N_vec = [[x for x in v] ./ norm(v) for v in N_disp]
NN_vec = [[x for x in v] ./ norm(v) for v in NN_disp]

self_coupling_N = [v * v' for v in N_vec] |> sum
self_coupling_NN = [v * v' for v in NN_vec] |> sum
self_coupling_matrix = k1 .* self_coupling_N + k2 .* self_coupling_NN
self_coupling =
    [Coupling((0, 0, 0), n, m, self_coupling_matrix[n, m]) for n = 1:3, m = 1:3] |> vec
self_coupling = filter(x -> x.k != 0, self_coupling)
couplings = vcat(N_couplings, NN_couplings, self_coupling)

# Make the system
sys = system(size_x, size_y, size_z, couplings)
println("Lattice relaxation")
# Relax the system and prepare initial conditions
rel = relaxation(sys, σ_init, λ, Φ0, nStep)
ρ_init = rel[5]
ρ_dot_init = zeros(length(sys[1]))
# Get all the atoms with which the particle interacts
atoms = Iterators.product([1:size_x, 1:size_y, 1:size_z]...) |> collect |> vec

atoms =
    Iterators.product(
        [
            floor(Int, size_x / 2)-5:floor(Int, size_x / 2)+5,
            floor(Int, size_y / 2)-5:floor(Int, size_y / 2)+5,
            floor(Int, size_z / 2)-5:floor(Int, size_z / 2)+5,
        ]...,
    ) |>
    collect |>
    vec
atom_indices = [
    [
        get(sys[1], c, ErrorException("Coordinate not found")) for
        c in [(a..., d) for d in [1, 2, 3]]
    ] for a in atoms
]

function lattice_response(sys, ρ_init, ρ_dot_init, σ_init, σ_dot_init)
    # Set up initial conditions
    σ = σ_init
    σ_dot = σ_dot_init
    # Time step depends on the speed to make sure we always get nPts steps
    δτ = ((σ_final-σ_init)[3]) / σ_dot[3] / nPts
    current_state = (ρ_init, ρ_dot_init, σ, σ_dot)
    # Preallocate arrays to save energies at each step
    interaction_energies = zeros(nPts + 1)
    elastic_energies = zeros(nPts + 1)
    lattice_kinetic_energies = zeros(nPts + 1)
    total_lattice_energies = zeros(nPts + 1)
    total_energies = zeros(nPts + 1)

    ρ = current_state[1]
    ρ_dot = current_state[2]
    σ = current_state[3]

    # Calculate the energies in the beginning of the calculation
    atom_positions = [ρ[atom_indices[a]] .+ (atoms[a] .- 1) .* α for a in eachindex(atoms)]

    interaction_energies[1] = [Φ(pos - σ) for pos in atom_positions] |> sum
    elastic_energies[1] = ρ' * sys[2] * ρ / 2
    lattice_kinetic_energies[1] = ρ_dot' * ρ_dot / 8 / pi^2
    total_lattice_energies[1] = elastic_energies[1] + lattice_kinetic_energies[1]
    total_energies[1] = interaction_energies[1] + total_lattice_energies[1]

    for jj = 1:nPts
        # Advance the system
        current_state = RKstep(μ, α, sys, atoms, Φ, current_state, δτ)
        ρ = current_state[1]
        ρ_dot = current_state[2]
        σ = current_state[3]
        # Calculate and save the energies
        atom_positions =
            [ρ[atom_indices[a]] .+ (atoms[a] .- 1) .* α for a in eachindex(atoms)]
        interaction_energies[jj+1] = [Φ(pos - σ) for pos in atom_positions] |> sum
        elastic_energies[jj+1] = ρ' * sys[2] * ρ / 2
        lattice_kinetic_energies[jj+1] = ρ_dot' * ρ_dot / 8 / pi^2
        total_lattice_energies[jj+1] =
            elastic_energies[jj+1] + lattice_kinetic_energies[jj+1]
        total_energies[jj+1] = interaction_energies[jj+1] + total_lattice_energies[jj+1]
    end
    # Get the maximum energies with respect to the initial relaxed configuration
    maximum_interaction = maximum(interaction_energies .- interaction_energies[1])
    maximum_elastic = maximum(elastic_energies .- elastic_energies[1])
    maximum_lattice_kinetic =
        maximum(lattice_kinetic_energies .- lattice_kinetic_energies[1])
    maximum_total_lattice = maximum(total_lattice_energies .- total_lattice_energies[1])
    maximum_total = maximum(total_energies .- total_energies[1])


    return (
        maximum_interaction,
        maximum_elastic,
        maximum_lattice_kinetic,
        maximum_total_lattice,
        maximum_total,
    )
end

println("Calculating...")

if !isfile("Data/LatticeResponse.jld2")
    res = Vector{Tuple{Float64,Float64,Float64,Float64,Float64}}(undef, nSpeeds)
    pr = Progress(nSpeeds)
    Threads.@threads for ii = 1:nSpeeds
        res[ii] = lattice_response(sys, ρ_init, ρ_dot_init, σ_init, [0, 0, speeds[ii]])
        next!(pr)
    end
    save_object("Data/LatticeResponse.jld2", res)
end


r = load_object("Data/LatticeResponse.jld2")

interaction = [x[1] for x in r]
elastic = [x[2] for x in r]
kinetic = [x[3] for x in r]
lattice = [x[4] for x in r]
total = [x[5] for x in r]

scatter(speeds, interaction)
scatter(speeds, elastic)
scatter(speeds, kinetic )
scatter(speeds, lattice)
scatter(speeds, total)
scatter(speeds, speeds.^2 ./ 8 ./ pi^2 ./ 10)
