using Distributions
using ForwardDiff
using JLD2
using LinearAlgebra
using ProgressMeter
using QuadGK
using SparseArrays

struct Coupling
    disp::Tuple{Int64,Int64,Int64}  # Displacement
    d1::Int64                       # Coordinate of the first atom
    d2::Int64                       # Coordinate of the second atom
    k::Float64                      # Coupling strength
end

function system(size_x::Int, size_y::Int, size_z::Int, coupling::Vector{Coupling})
    # Enumerate all the atoms in the system
    atoms = Iterators.product([1:size_x, 1:size_y, 1:size_z]...) |> collect |> vec
    # Enumerate all the coordinates in the system
    coords = [(a..., c) for c in [1, 2, 3], a in atoms] |> vec

    # Assign each coordinate an ordinal index
    index = Dict(zip(coords, 1:length(coords)))

    coupling_elements = vcat(
        [
            [
                (
                    # For each atom, take the cartesian coordinate from the couping term and get the corresponding index
                    get(index, (a..., c.d1), nothing),
                    # Using the displacement from the coupling, 
                    # get the atom to which "a" couples, as well as its cartesian coordinate (c.d2 field)
                    # To make the system periodic, let b = a + c.disp, the lattice coordinate of the target atom
                    # Atoms are enumerated from 1 to size_{x/y/z}. For periodicity, we use modulo, requiring
                    # enumeration from 0 to size_{x/y/z} - 1. Hence, we subtract (1,1,1) from b before applying
                    # the modulo and then add (1,1,1) after.
                    get(
                        index,
                        (
                            (1, 1, 1) .+
                            mod.(a .+ c.disp .- (1, 1, 1), (size_x, size_y, size_z))...,
                            c.d2,
                        ),
                        nothing,
                    ),
                    # Set the coupling between the two indices to the magnitude from c
                    c.k,
                ) for c in coupling
            ] for a in atoms
        ]...,
    )
    row = [c[1] for c in coupling_elements]
    col = [c[2] for c in coupling_elements]
    val = [c[3] for c in coupling_elements]
    # Assemble to coupling tuples into a sparse matrix
    Ξ = sparse(row, col, val)
    return (index, Ξ)
end
