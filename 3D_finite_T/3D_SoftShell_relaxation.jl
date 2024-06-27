include("../src/main.jl")
include("../src/plotting.jl")

## PARAMETERS
μ = [Inf, Inf, 1 / 10]      # Mass of the particle in the units of chain masses
α = 5                       # Lattice constant
k1 = 15                     # Nearest neighbor force constant
k2 = 5                      # Next-nearest neighbor force constant

size_x = size_y = size_z = 20
# Interaction
Φ0 = 5000                   # Amplitude of the Yukawa interaction
λ = 1.5

# @inline function Φ(r)
#     res = Φ0 / (1 + exp((norm(r) - 3) / λ))
#     # res = Φ0 / (1 + exp((norm(r) - 5 * sqrt(2) / 2) / λ))
#     return res
# end
2.5 * sqrt(2)
# Interaction function
@inline function Φ(r)
    res = Φ0 * exp(-dot(r, r) / 2 / λ^2)
    return res
end

# @inline function Φ(r)
#     res = Φ0 * exp(-norm(r)  / λ) / norm(r)
#     return res
# end

atom_positions_0 = [(atoms[a] .- 1) .* α for a in eachindex(atoms)]

Pot = [Φ(pos .- σ_mid) for pos in atom_positions_0] |> sum
Pot = [Φ(pos .- σ_edge) for pos in atom_positions_0] |> sum


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
atoms = Iterators.product([1:size_x, 1:size_y, 1:size_z]...) |> collect |> vec
atom_indices = [
    [
        get(sys[1], c, ErrorException("Coordinate not found")) for
        c in [(a..., d) for d in [1, 2, 3]]
    ] for a in atoms
]

σ_mid = [(size_x + 1) / 2 - 1, (size_y + 1) / 2 - 1, (size_z + 1) / 2 - 1] .* α
σ_edge = [(size_x + 0) / 2 - 1, (size_y + 1) / 2 - 1, (size_z + 1) / 2 - 1] .* α
σ_dot = [0, 0, 0]

ρ = zeros(length(sys[1]))
ρ_dot = zeros(length(sys[1]))
for ii = 1:300
    atom_positions = [ρ[atom_indices[a]] .+ (atoms[a] .- 1) .* α for a in eachindex(atoms)]
    rr = derivatives(μ, α, sys..., atoms, Φ, ρ, ρ_dot, σ_mid, σ_dot)

    ρ += 0.01 .* normalize(rr[2])
end
atom_positions = [ρ[atom_indices[a]] .+ (atoms[a] .- 1) .* α for a in eachindex(atoms)]

Pot = [Φ(pos - σ_mid) for pos in atom_positions] |> sum
elastic = ρ' * sys[2] * ρ / 2
Pot + elastic


ρ = zeros(length(sys[1]))
ρ[1] = 1.0
elastic = ρ' * sys[2] * ρ / 2



maximum(abs.(ρ))

ρ = zeros(length(sys[1]))
ρ_dot = zeros(length(sys[1]))
for ii = 1:300
    atom_positions = [ρ[atom_indices[a]] .+ (atoms[a] .- 1) .* α for a in eachindex(atoms)]
    rr = derivatives(μ, α, sys..., atoms, Φ, ρ, ρ_dot, σ_edge, σ_dot)
    ρ += 0.01 .* normalize(rr[2])
end

atom_positions = [ρ[atom_indices[a]] .+ (atoms[a] .- 1) .* α for a in eachindex(atoms)]

Pot = [Φ(pos - σ_edge) for pos in atom_positions] |> sum
elastic = ρ' * sys[2] * ρ / 2
Pot + elastic





equilibrium_positions = [(atoms[a] .- 1) .* α for a in eachindex(atoms)]
displacement = [norm(p) for p in [ρ[atom_indices[a]] for a in eachindex(atoms)]]
set_theme!(CF_theme)
# cmap_alpha = resample_cmap(CF_heat, n; alpha = alphas)

fig = Figure()


ax1 = Axis3(fig[1, 1], aspect = :equal, title = "aspect = :equal")

scatter!(
    ax1,
    [p[1] for p in equilibrium_positions],
    [p[2] for p in equilibrium_positions],
    [p[3] for p in equilibrium_positions],
    color = displacement,
    markersize = 30,
    colormap = :vik,
    alpha = 0.5,
)
xlims!(ax1, (40, 50))
ylims!(ax1, (40, 50))
zlims!(ax1, (40, 50))
fig
# maximum(displacement)



s

dyn_mat = dynamical_matrix(couplings)

cr2_5 = Corr(dyn_mat, 2.5, 0, [0, 0, 0])[1]
cr5 = Corr(dyn_mat, 5, 0, [0, 0, 0])[1]
sqrt.(abs.(cr2_5))
sqrt.(abs.(cr5))

cr2_5


# maximum(abs.(rr[2]))
maximum(abs.(ρ))
atoms[3790]

ρ[get(sys[1], (10, 10, 10, 2), nothing)]
ρ[get(sys[1], (10, 10, 10, 3), nothing)]

ρ[get(sys[1], (10, 10, 11, 2), nothing)]
ρ[get(sys[1], (10, 10, 11, 3), nothing)]

ρ[get(sys[1], (10, 11, 10, 2), nothing)]
ρ[get(sys[1], (10, 11, 10, 3), nothing)]

ρ[get(sys[1], (10, 11, 11, 2), nothing)]
ρ[get(sys[1], (10, 11, 11, 3), nothing)]


rr = α / 2 * (sqrt(3) - sqrt(2)) / sqrt(2)
ρ[get(sys[1], (10, 10, 10, 2), nothing)] = -rr
ρ[get(sys[1], (10, 10, 10, 3), nothing)] = -rr

ρ[get(sys[1], (10, 10, 11, 2), nothing)] = -rr
ρ[get(sys[1], (10, 10, 11, 3), nothing)] = rr

ρ[get(sys[1], (10, 11, 10, 2), nothing)] = rr
ρ[get(sys[1], (10, 11, 10, 3), nothing)] = -rr

ρ[get(sys[1], (10, 11, 11, 2), nothing)] = rr
ρ[get(sys[1], (10, 11, 11, 3), nothing)] = rr
ρ' * sys[2] * ρ / 2
# get(sys[1], (11,11,11,3),nothing)
15 / 2 * ((rr.*[1, 1]-rr*[1, -1])[2])^2

res = load_object("Data/Relaxation.jld2")
res[2]

Base.summarysize(res)
