include("../src/main.jl")
include("../src/plotting.jl")

## PARAMETERS
μ = [Inf, Inf, 1 / 10]      # Mass of the particle in the units of chain masses
α = 5                       # Lattice constant
k1 = 15                     # Nearest neighbor force constant
k2 = 5                      # Next-nearest neighbor force constant

size_x = size_y = size_z = 50
atoms = Iterators.product([1:size_x, 1:size_y, 1:size_z]...) |> collect |> vec

speed_max = (2 * π * α * √(k1 + 2 * k2)) / 2  # Max speed is 0.5 speed of sound
δτ = 1e-3
# Interaction
Φ0 = 1 / 100
λs = [1, 1 / 2, 1 / 4, 1 / 8]

# Interaction
nPts = 50

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

# Dynamical matrix
DynamicalMatrixSmall = dynamical_matrix_small(couplings)

# Make the system
sys = system(size_x, size_y, size_z, couplings)

println("Starting calculations...")

## SIMULATIONS
face_atoms = [(1, 1, 1), (1, 2, 1), (2, 1, 1), (2, 2, 1)]
cube_atoms =
    [(1, 1, 1), (1, 2, 1), (2, 1, 1), (2, 2, 1), (1, 1, 2), (1, 2, 2), (2, 1, 2), (2, 2, 2)]

for λ in λs

    @inline function Φ(r)
        res = Φ0 / (1 + exp((norm(r) - 3) / λ))
        return (isnan(res) ? 0 : res)
    end

    σ_edge = [floor(size_x / 2), floor(size_y / 2) + 1 / 2, floor(size_z / 2) + 1 / 2] .* α

    atom_positions = [(atoms[a] .- 1) .* α for a in eachindex(atoms)]
    Pot = [Φ(pos .- σ_edge) for pos in atom_positions] |> sum

    speed_min = (Pot * 8 * π^2 / μ[3] |> sqrt) * 2
    init_speeds = range(speed_min, speed_max, length = nPts)

    if !isfile("Data/3D_Loss/3D_SoftShell_Square_Loss_λ$(λ).jld2")

        loss = zeros(nPts)
        prog = Progress(nPts)
        Threads.@threads for ii in eachindex(loss)
            init_pos = -3 * α
            init_speed = init_speeds[ii]
            println(init_speed)
            σ = [α / 2, α / 2, init_pos]
            σ_dot = [0, 0, init_speed]

            ρ_init = zeros(length(sys[1]))
            ρ_dot_init = zeros(length(sys[1]))

            current_state = (ρ_init, ρ_dot_init, σ, σ_dot)
            while current_state[3][3] < 3 * α
                current_state = RKstep(μ, α, sys, face_atoms, Φ, current_state, δτ)
                GC.safepoint()
            end
            final_speed = current_state[4][3]
            loss[ii] = ((init_speed^2) / (8 * π^2) - (final_speed^2) / (8 * π^2)) * μ[3]
            next!(prog)
        end

        save_object(
            "Data/3D_Loss/3D_SoftShell_Square_Loss_λ$(λ).jld2",
            (λ, init_speeds, loss),
        )
    end

    if !isfile("Data/3D_Loss/3D_SoftShell_Cube_Loss_λ$(λ).jld2")

        loss = zeros(nPts)
        prog = Progress(nPts)
        Threads.@threads for ii in eachindex(loss)
            init_pos = -3 * α
            init_speed = init_speeds[ii]
            σ = [α / 2, α / 2, init_pos]
            σ_dot = [0, 0, init_speed]

            ρ_init = zeros(length(sys[1]))
            ρ_dot_init = zeros(length(sys[1]))

            current_state = (ρ_init, ρ_dot_init, σ, σ_dot)
            while current_state[3][3] < 4 * α
                current_state = RKstep(μ, α, sys, cube_atoms, Φ, current_state, δτ)
                GC.safepoint()
            end
            final_speed = current_state[4][3]
            loss[ii] = ((init_speed^2) / (8 * π^2) - (final_speed^2) / (8 * π^2)) * μ[3]
            next!(prog)
        end

        save_object(
            "Data/3D_Loss/3D_SoftShell_Cube_Loss_λ$(λ).jld2",
            (λ, init_speeds, loss),
        )
    end

end


# init_pos = -3 * α
# init_speed = 60
# println(init_speed)
# σ = [α / 2, α / 2, init_pos]
# σ_dot = [0, 0, init_speed]

# ρ_init = zeros(length(sys[1]))
# ρ_dot_init = zeros(length(sys[1]))

# current_state = (ρ_init, ρ_dot_init, σ, σ_dot)
# while current_state[3][3] < 3 * α
#     current_state = RKstep(μ, α, sys, face_atoms, Φ, current_state, δτ)
#     println(current_state[3][3])
# end
# final_speed = current_state[4][3]
# ((init_speed^2) / (8 * π^2) - (final_speed^2) / (8 * π^2)) * μ[3]
# λ = 1/2
# @inline function Φ(r)
#     res = Φ0 / (1 + exp((norm(r) - 3) / λ))
#     return (isnan(res) ? 0 : res)
# end
