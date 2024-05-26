using ForwardDiff
# using HCubature
using JLD2
# using LinearAlgebra
using ProgressMeter
using QuadGK
# using SparseArrays
# using SpecialFunctions
include("system.jl")
include("analytic.jl")
# ## PREPARE DIRECTORIES
# dirs = ["Data", "Data/1D", "Data/3D"]
# [isdir(d) ? nothing : mkdir(d) for d in dirs]

# struct Coupling
#     disp::Tuple{Int64,Int64,Int64}  # Displacement
#     d1::String                      # Coordinate of the first atom
#     d2::String                      # Coordinate of the second atom
#     k::Float64                      # Coupling strength
# end

# function system(size_x::Int, size_y::Int, size_z::Int, coupling::Vector{Coupling})
#     # Enumerate all the atoms in the system
#     atoms = [(nx, ny, nz) for nx = 1:size_x, ny = 1:size_y, nz = 1:size_z] |> vec
#     # Enumerate all the coordinates in the system
#     coords = [(a..., c) for c in ["x", "y", "z"], a in atoms] |> vec

#     # Assign each coordinate an ordinal index
#     index = Dict(zip(coords, 1:length(coords)))

#     coupling_elements = vcat(
#         [
#             [
#                 (
#                     # For each atom, take the cartesian coordinate from the couping term and get the corresponding index
#                     get(index, (a..., c.d1), nothing),
#                     # Using the displacement from the coupling, 
#                     # get the atom to which "a" couples, as well as its cartesian coordinate (c.d2 field)
#                     # To make the system periodic, let b = a + c.disp, the lattice coordinate of the target atom
#                     # Atoms are enumerated from 1 to size_{x/y/z}. For periodicity, we use modulo, requiring
#                     # enumeration from 0 to size_{x/y/z} - 1. Hence, we subtract (1,1,1) from b before applying
#                     # the modulo and then add (1,1,1) after.
#                     get(
#                         index,
#                         (
#                             (1, 1, 1) .+
#                             mod.(a .+ c.disp .- (1, 1, 1), (size_x, size_y, size_z))...,
#                             c.d2,
#                         ),
#                         nothing,
#                     ),
#                     # Set the coupling between the two indices to the magnitude from c
#                     c.k,
#                 ) for c in coupling
#             ] for a in atoms
#         ]...,
#     )
#     row = [c[1] for c in coupling_elements]
#     col = [c[2] for c in coupling_elements]
#     val = [c[3] for c in coupling_elements]
#     # Assemble to coupling tuples into a sparse matrix
#     Ξ = sparse(row, col, val)
#     return (index, Ξ)
# end

# function derivatives(μ, α, index, Ξ, atoms, Φ, ρ, ρdot, σ, σdot)
#     # Intrinsic framework acceleration
#     ddρ = -4 * π^2 * Ξ * ρ

#     # Get the indices of the coordinates of each atom in 'atoms' interacting with the mobile particle
#     atom_indices = [
#         [
#             get(index, c, ErrorException("Coordinate not found")) for
#             c in [(a..., d) for d in ["x", "y", "z"]]
#         ] for a in atoms
#     ]

#     # Get the positions of each atom in 'atoms'. The first atom is at (0,0,0)
#     atom_positions = [ρ[atom_indices[a]] .+ (atoms[a] .- 1) .* α for a in eachindex(atoms)]

#     # Calculate the negative gradient of the interaction with each atom in 'atoms' and multiply its
#     # by 4π² to get the force

#     F_ρ = [-4 * π^2 .* ForwardDiff.gradient(Φ, pos - σ) for pos in atom_positions]
#     # Force on mobile particles due to interaction using Newton's 3d law
#     F_σ = -sum(F_ρ)

#     # Add the force to ddρ to get the full acceleration (atom mass = 1, so no mass division)
#     view(ddρ, vcat(atom_indices...)) .+= vcat(F_ρ...)

#     return (ρdot, ddρ, σdot, F_σ ./ μ)
# end

# function RKstep(μ, α, sys, atoms, Φ, current_state, δτ)
#     k1 = derivatives(μ, α, sys..., atoms, Φ, current_state...)
#     k2 = derivatives(μ, α, sys..., atoms, Φ, (current_state .+ k1 .* (δτ / 4))...)
#     k3 = derivatives(μ, α, sys..., atoms, Φ, (current_state .+ (k1 .+ k2) .* (δτ / 8))...)
#     k4 = derivatives(
#         μ,
#         α,
#         sys...,
#         atoms,
#         Φ,
#         (current_state .+ k3 .* δτ .- k2 .* (δτ / 2))...,
#     )
#     k5 = derivatives(
#         μ,
#         α,
#         sys...,
#         atoms,
#         Φ,
#         (current_state .+ k1 .* (δτ * 3 / 16) .+ k4 .* (δτ * 9 / 16))...,
#     )
#     k6 = derivatives(
#         μ,
#         α,
#         sys...,
#         atoms,
#         Φ,
#         (
#             current_state .- k1 .* (3 / 7 * δτ) .+ k2 .* (2 / 7 * δτ) .+
#             k3 .* (12 / 7 * δτ) .- k4 .* (12 / 7 * δτ) .+ k5 .* (8 / 7 * δτ)
#         )...,
#     )
#     res =
#         current_state .+
#         (δτ / 90) .* (7 .* k1 .+ 32 .* k3 .+ 12 .* k4 .+ 32 .* k5 .+ 7 .* k6)
#     return res
# end

# # Calculate the loss for a single pass when the particle moves with a constant speed
# # in the z-direction with impact parameter s and the width/strength of the Gaussian
# # interaction is given by λ/Φ0
# function numerical_Δ(Φ0, λ, s, σ_dot, DynamicalMatrix, dims)
#     function fun_int(q)
#         # Solve the eigenproblem
#         eig = eigen(DynamicalMatrix(q...))
#         ωs = sqrt.(eig.values)
#         ηs = eig.vectors
#         # Displacement vector
#         σ0 = [0, s, 0]

#         potential_gradient = [
#             √(2 * π) *
#             λ *
#             (σ0 ./ λ^2 + 2im * π * [0, 0, ω] ./ σ_dot) *
#             exp((-dot(σ0, σ0) - 4 * π^2 * λ^4 * ω^2 / σ_dot^2) / 2 / λ^2) for ω in ωs
#         ]
#         mode_coupling = [abs(dot(ηs[:, jj], potential_gradient[jj]) / σ_dot) for jj = 1:3]
#         res = dot(mode_coupling, mode_coupling) * 2 * π^2 * Φ0^2
#         return res
#     end
#     res = hcubature(
#         q -> fun_int(q) / (2 * π)^dims,
#         0.0 .* ones(dims),
#         2.0 .* π .* ones(dims),
#         rtol = 1e-3,
#         initdiv = 30,
#     )
#     return res
# end

# function Loss_Matrix(DynamicalMatrix_Small)
#     function fun_int(q)
#         # Solve the eigenproblem
#         eig = eigen(DynamicalMatrix_Small(q...))
#         ωs = sqrt.(eig.values)
#         ηs = eig.vectors
#         res = sum([(ηs[:, jj] * ηs[:, jj]') ./ ωs[jj]^3 for jj = 1:3])
#         return res
#     end
#     res = hcubature(
#         q -> sin(q[1]) .* fun_int(q) / 32 / π^3,
#         [0, 0],
#         [π, 2 * π],
#         rtol = 1e-3,
#         # initdiv = 20,
#     )
#     return res
# end
