include("../src/main.jl")
include("../src/plotting.jl")

## PARAMETERS
μ = [Inf, Inf, 1 / 10]      # Mass of the particle in the units of chain masses
α = 5                       # Lattice constant
k1 = 150                    # Nearest neighbor force constant
k2 = 50                      # Next-nearest neighbor force constant

size_x = size_y = size_z = 30

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

DynMat = dynamical_matrix(couplings)
ωT = 25
Corr(DynMat, ωT, 0.5, [3, 3, 3])[1] ./ ωT
Corr(DynMat, ωT, 0.0, [0, 0, 0])[1]
0.08 / 10 * 3 |> sqrt

# Make the system
sys = system(size_x, size_y, size_z, couplings)

σ_mid = [(size_x + 1) / 2 - 1, (size_y + 1) / 2 - 1, (size_z + 1) / 2 - 1] .* α
σ_edge = [(size_x + 0) / 2 - 1, (size_y + 1) / 2 - 1, (size_z + 1) / 2 - 1] .* α
σ_dot = [0, 0, 0]


## RELAXATION
function relaxation(sys, σ, λ, Φ0, nStep)
    atoms = Iterators.product([1:size_x, 1:size_y, 1:size_z]...) |> collect |> vec
    atom_indices = [
        [
            get(sys[1], c, ErrorException("Coordinate not found")) for
            c in [(a..., d) for d in [1, 2, 3]]
        ] for a in atoms
    ]
    @inline function Φ(r)
        res = Φ0 * exp(-dot(r, r) / 2 / λ^2)
        return res
    end

    ρ = zeros(length(sys[1]))
    ρ_dot = zeros(length(sys[1]))
    if nStep > 0

        for _ = 1:nStep
            d = derivatives(μ, α, sys..., atoms, Φ, ρ, ρ_dot, σ, [0, 0, 0])
            step = min(0.1, maximum(abs.(d[2])))
            ρ += step .* normalize(d[2])
        end

        for _ = 1:nStep
            d = derivatives(μ, α, sys..., atoms, Φ, ρ, ρ_dot, σ, [0, 0, 0])
            step = min(0.01, maximum(abs.(d[2])))
            ρ += step .* normalize(d[2])
        end

        for _ = 1:nStep
            d = derivatives(μ, α, sys..., atoms, Φ, ρ, ρ_dot, σ, [0, 0, 0])
            step = min(0.001, maximum(abs.(d[2])))
            ρ += step .* normalize(d[2])
        end

        for _ = 1:nStep
            d = derivatives(μ, α, sys..., atoms, Φ, ρ, ρ_dot, σ, [0, 0, 0])
            step = min(0.0001, maximum(abs.(d[2])))
            ρ += step .* normalize(d[2])
        end

        for _ = 1:nStep
            d = derivatives(μ, α, sys..., atoms, Φ, ρ, ρ_dot, σ, [0, 0, 0])
            step = min(0.00001, maximum(abs.(d[2])))
            ρ += step .* normalize(d[2])
        end

    end
    atom_positions = [ρ[atom_indices[a]] .+ (atoms[a] .- 1) .* α for a in eachindex(atoms)]
    interaction = [Φ(pos - σ) for pos in atom_positions] |> sum
    elastic = ρ' * sys[2] * ρ / 2
    max_disp = maximum(abs.(ρ))
    d = derivatives(μ, α, sys..., atoms, Φ, ρ, ρ_dot, σ, [0, 0, 0])
    return (interaction, elastic, max_disp, d[2], ρ)
end

λ = α / 4
Φ0 = 5000

m_unrelaxed = relaxation(sys, σ_mid, λ, Φ0, 0)
e_unrelaxed = relaxation(sys, σ_edge, λ, Φ0, 0)

diff_unrelaxed = sum(e_unrelaxed[1:2]) - sum(m_unrelaxed[1:2])
m_unrelaxed[4] |> maximum

m_relaxed = relaxation(sys, σ_mid, λ, Φ0, 150)
e_relaxed = relaxation(sys, σ_edge, λ, Φ0, 150)

diff_relaxed = sum(e_relaxed[1:2]) - sum(m_relaxed[1:2])

# abs.(m_relaxed[end]) |> maximum
m_unrelaxed[end]

sum(e_relaxed[1]) - sum(m_relaxed[1])
sum(e_relaxed[2]) - sum(m_relaxed[2])


m_relaxed[4] |> maximum
e_relaxed[4] |> maximum


atoms = Iterators.product([1:size_x, 1:size_y, 1:size_z]...) |> collect |> vec
atom_indices = [
    [
        get(sys[1], c, ErrorException("Coordinate not found")) for
        c in [(a..., d) for d in [1, 2, 3]]
    ] for a in atoms
]

@inline function Φ(r)
    res = Φ0 * exp(-dot(r, r) / 2 / λ^2)
    return res
end
atom_positions = [ρ[atom_indices[a]] .+ (atoms[a] .- 1) .* α for a in eachindex(atoms)]

dot(vcat([ForwardDiff.gradient(Φ, pos - σ) for pos in atom_positions]...), m_relaxed[end])
dot(vcat([ForwardDiff.gradient(Φ, pos - σ) for pos in atom_positions]...), e_relaxed[end])


dot(vcat([ForwardDiff.hessian(Φ, pos - σ) for pos in atom_positions]...), e_relaxed[end])
[ForwardDiff.hessian(Φ, pos - σ) for pos in atom_positions]
1
dot(e_relaxed[end], m_relaxed[end])

σ_mid
RKstep(μ, α, sys, atoms, Φ, current_state, δτ)

# ## DATA COLLECTION
# nPts = 100
# nSteps = 300

# λmin = 1 / 10
# λmax = 3
# λs = range(λmin, λmax, length = nPts)

# Φmin = 1
# Φmax = 6000
# Φs = range(Φmin, Φmax, length = nPts)

# res_mid = Matrix{Tuple{Float64,Float64,Float64}}(undef, nPts, nPts)
# res_edge = Matrix{Tuple{Float64,Float64,Float64}}(undef, nPts, nPts)
# pr = Progress(nPts * nPts)
# if !isfile("Data/Relaxation.jld2")
#     for ii = 1:nPts
#         λ = λs[ii]
#         Threads.@threads for jj = 1:nPts
#             Φ0 = Φs[jj]
#             res_mid[ii, jj] = relaxation(sys, σ_mid, λ, Φ0, nSteps)
#             res_edge[ii, jj] = relaxation(sys, σ_edge, λ, Φ0, nSteps)
#             next!(pr)
#         end
#     end
#     save_object("Data/Relaxation.jld2", (res_mid, res_edge))
# end


# res = load_object("Data/Relaxation.jld2")

# mid = res[1]
# edge = res[2]

# mid_energy = [x[1] + x[2] for x in mid]
# edge_energy = [x[1] + x[2] for x in edge]
# contour(λs, Φs, edge_energy)
# heatmap(λs, Φs, mid_energy - edge_energy)

# minimum(filter(x -> !isnan(x), mid_energy - edge_energy))



# Set up initial conditions
σ = σ_mid
σ_dot = [0, 0, 0]
# Time step depends on the speed to make sure we always get nPts steps
δτ = 0.01
current_state = (m_unrelaxed[end], m_unrelaxed[end], σ, σ_dot)
# Preallocate arrays to save energies at each step


current_state = RKstep([Inf, Inf, Inf], α, sys, atoms, Φ, current_state, δτ)

norm(current_state[1] - m_relaxed[end])



m_unrelaxed[end]
σ_dot
