include("../src/main.jl")

dirs = ["Data", "Data/3D_Loss"]
[isdir(d) ? nothing : mkdir(d) for d in dirs]
## PARAMETERS
μ = [Inf, Inf, 1]           # Mass of the particle in the units of chain masses
α = 2                       # Lattice constant
k1 = 15                     # Nearest neighbor force constant
k2 = 5                      # Next-nearest neighbor force constant

size_x = size_y = size_z = 50
# Interaction
Φ0 = 0.01                   # Amplitude of the Gaussian interaction
s_vals = [0, 1]             # Impact parameters
λ_vals = [1 / 2, 1, 2, 4]   # Extent of the Gaussian interaction
params = [(s, λ) for s in s_vals, λ in λ_vals] |> vec

nPts = 50
speed_min = 1
speed_max = 80
init_speeds = range(speed_min, speed_max, length = nPts)

# Simulation settings
nPts = 50                   # Number of points
δτ = 1e-3                   # Time step

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
DynamicalMatrix = dynamical_matrix(couplings)
DynamicalMatrixSmall = dynamical_matrix_small(couplings)

# Make the system
sys = system(size_x, size_y, size_z, couplings)
atoms = [(1, 1, 1)]           # The moving particle interacts with a single atom

function Φ_full(σ, λ, atoms, α)
    res = [exp(-norm(σ .- (a .- (1, 1, 1)) .* α)^2 / 2 / λ^2) for a in atoms] |> sum
    return res
end

function Hess_Φ(σ, λ, atoms, α)
    res = ForwardDiff.hessian(σ -> Φ_full(σ, λ, atoms, α), σ)
    return res
end
M_loss = Loss_Matrix(DynamicalMatrixSmall)[1]

# Δ/(σ̇ Φ0²) in low-speed limit
function Δ_pass(σ0, σ_dot, λ, atoms, α)
    σ_dot_unit = normalize(σ_dot)
    r = quadgk(
        t ->
            σ_dot_unit' *
            Hess_Φ(σ_dot_unit * t + σ0, λ, atoms, α) *
            M_loss *
            Hess_Φ(σ_dot_unit * t + σ0, λ, atoms, α) *
            σ_dot_unit,
        -Inf,
        Inf,
    )
    return r[1]

end

println("Starting calculations...")

## SIMULATIONS
for p in params
    s = p[1]
    λ = p[2]

    if !isfile("Data/3D_Loss/3D_Gaussian_Loss_s$(s)_λ$(λ).jld2")

        # Interaction function
        @inline function Φ(r)
            res = Φ0 * exp(-dot(r, r) / 2 / λ^2)
            return res
        end

        loss = zeros(nPts)
        numerical_loss = zeros(nPts)
        prog = Progress(nPts)
        Threads.@threads for ii in eachindex(loss)
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
            next!(prog)
        end

        save_object(
            "Data/3D_Loss/3D_Gaussian_Loss_s$(s)_λ$(λ).jld2",
            (s, λ, init_speeds, loss, numerical_loss),
        )
    end
end
