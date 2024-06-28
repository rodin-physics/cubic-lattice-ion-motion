include("../src/unitful.jl")

## PARAMETERS
a = 3                       # Lattice constant in Å
k1 = 520                    # Nearest neighbor force constant in meV / Å²
k2 = 170                    # Next-nearest neighbor force constant in meV / Å²
m = 3.5                     # Lattice mass in meV * (ps / Å)²
ħ = 0.6582119569            # Planck constant in meV * ps

dt = 1e-2
t_max = 2
nPts = floor(t_max / dt) |> Int

size_x = size_y = size_z = 100

λ = a / 4
U0 = 40000

@inline function U(r)
    res = U0 * exp(-dot(r, r) / 2 / λ^2)
    return res
end

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
DynamicalMatrix = dynamical_matrix(m, couplings)
DynamicalMatrixSmall = dynamical_matrix_small(a, m, couplings)
LossMat = Loss_Matrix(a, m, DynamicalMatrixSmall)[1]
W(t) = Recoil(m, DynamicalMatrix, t, [0,0,0])[1]
V = sys[2][1,1]
# Make the system
sys = system(size_x, size_y, size_z, couplings)
println("Starting calculations...")



















## SIMULATIONS
# Compute the particle trajectory using the full system

pos = Vector{Vector{Float64}}(undef, nPts)
speed = Vector{Vector{Float64}}(undef, nPts)
pos[1] = σ
speed[1] = σ_dot
ρ_init = zeros(length(sys[1]))
ρ_dot_init = zeros(length(sys[1]))
function solve_full()
    current_state = (ρ_init, ρ_dot_init, σ, σ_dot)

    @showprogress for ii = 2:nPts
        current_state = RKstep(μ, α, sys, atoms, Φ, current_state, δτ)
        pos[ii] = current_state[3]
        speed[ii] = current_state[4]
    end
    return (pos, speed)
end

M_matrix = (Loss_Matrix(DynamicalMatrixSmall)[1])[3, 3] |> real

function polaron_potential(σ)
    res = Φ_total_grad(σ)' * polaron_matrix * Φ_total_grad(σ)
    return res
end

function total_potential(σ)
    res = Φ_total(σ) - polaron_potential(σ)
    return res
end

# -ForwardDiff.gradient(total_potential, [1,2,1])
# total_potential([0,0,3α])-total_potential([0,0,3.5α])
function derivative_particle(σ, σdot)
    F_potential = -ForwardDiff.gradient(total_potential, σ)

    Hess = ForwardDiff.derivative(
        y -> ForwardDiff.derivative(x -> Φ_total([σ[1], σ[2], x]), y),
        σ[3],
    )
    friction = -Hess^2 * M_matrix * σdot

    res = (σdot, 4 * π^2 * (F_potential + friction) ./ μ)
    return res
end

function RKstep_particle(current_state)
    k1 = derivative_particle(current_state...)
    k2 = derivative_particle((current_state .+ k1 .* (δτ / 4))...)
    k3 = derivative_particle((current_state .+ (k1 .+ k2) .* (δτ / 8))...)
    k4 = derivative_particle((current_state .+ k3 .* δτ .- k2 .* (δτ / 2))...)
    k5 = derivative_particle(
        (current_state .+ k1 .* (δτ * 3 / 16) .+ k4 .* (δτ * 9 / 16))...,
    )
    k6 = derivative_particle(
        (
            current_state .- k1 .* (3 / 7 * δτ) .+ k2 .* (2 / 7 * δτ) .+
            k3 .* (12 / 7 * δτ) .- k4 .* (12 / 7 * δτ) .+ k5 .* (8 / 7 * δτ)
        )...,
    )
    res =
        current_state .+
        (δτ / 90) .* (7 .* k1 .+ 32 .* k3 .+ 12 .* k4 .+ 32 .* k5 .+ 7 .* k6)
    return res
end

pos_particle = Vector{Vector{Float64}}(undef, nPts)
speed_particle = Vector{Vector{Float64}}(undef, nPts)
pos_particle[1] = σ
speed_particle[1] = σ_dot
function solve_particle()
    current_state = (σ, σ_dot)
    @showprogress for ii = 2:nPts
        current_state = RKstep_particle(current_state)
        pos_particle[ii] = current_state[1]
        speed_particle[ii] = current_state[2]
    end
    return (pos_particle, speed_particle)

end
if !isfile("Data/3D_Loss/Local_Loss.jld2")
    pos_particle, speed_particle = solve_particle()
end
if !isfile("Data/3D_Loss/Local_Loss.jld2")
    pos, speed = solve_full()
end

if !isfile("Data/3D_Loss/Local_Loss.jld2")
    save_object(
        "Data/3D_Loss/Local_Loss.jld2",
        (pos, speed, pos_particle, speed_particle, δτ, Φ0, λ),
    )
end
