include("../src/main.jl")

dirs = ["Data", "Data/1D_Diffusion"]
[isdir(d) ? nothing : mkdir(d) for d in dirs]
## PARAMETERS
μ = [Inf, Inf, 1]       # Mass of the particle in the units of chain masses
α = 2                   # Lattice constant
ω_perp = sqrt(99 / 4)   # Frequency of the chain spring for transverse mass displacements
ω_par = sqrt(99 / 4)    # Frequency of the chain spring for longitudinal mass displacements
ν = 1

ωT = 2

size_x = 1
size_y = 1
size_z = 200

# Interaction
Φ0 = 1                          # Amplitude of the Gaussian interaction
# s_vals = [0, 1]             # Impact parameters
# λ_vals = [1 / 2, 1, 2, 4]   # Extent of the Gaussian interaction
s = 0
λ = 1
# Simulation settings
nPts = 50                   # Number of points
δτ = 1e-3                   # Time step

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
sys = system(size_x, size_y, size_z, coupling)

# Dynamical matrix
DynamicalMatrix = dynamical_matrix(coupling)

qs =
    [
        2 .* π .* (qx / size_x, qy / size_y, qz / size_z) for qx = 0:(size_x-1),
        qy = 0:(size_y-1), qz = 0:(size_z-1)
    ] |> vec

normal_modes = [
    homogeneous_motion(DynamicalMatrix(q), ωT) ./ sqrt(size_x * size_y * size_z) for q in qs
]

normal_mode_pos = [n[1] for n in normal_modes]
normal_mode_speed = [n[2] for n in normal_modes]

atoms = Iterators.product([1:size_x, 1:size_y, 1:size_z]...) |> collect |> vec
ρH = zeros(ComplexF64, length(atoms) .* 3)
ρ_dotH = zeros(ComplexF64, length(atoms) .* 3)

for ii in eachindex(qs)
    phase = exp.(1im .* [dot(qs[ii], a) for a in atoms])
    ρH += kron(phase, normal_mode_pos[ii])
    ρ_dotH += kron(phase, normal_mode_speed[ii])
end

ρH = real(ρH)
ρ_dotH = real(ρ_dotH)

println("Starting calculations...")
# Interaction function
@inline function Φ(r)
    res = Φ0 * exp(-norm(r) / λ) ./ norm(r)
    return res
end

init_pos = 100 * α
σ = [0, s, init_pos]
σ_dot = [0, 0, 10]

ρ_init = ρH
ρ_dot_init = ρ_dotH
nPts = 100000
σs = zeros(nPts)
σs_dot = zeros(nPts)
σs[1] = init_pos
σs_dot[1] = 10
current_state = (ρ_init, ρ_dot_init, σ, σ_dot)
init_pos
@showprogress for ii = 2:nPts
    current_state = RKstep(μ, α, sys, atoms, Φ, current_state, δτ)
    σs[ii] = (current_state[3])[3]
    σs_dot[ii] = (current_state[4])[3]
end
σs
lines(σs)
lines(σs_dot)

σs_dot[10000:end] .^ 2 ./ 8 ./ pi^2 |> mean
