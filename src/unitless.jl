include("system.jl")
function derivatives(μ, α, index, Ξ, atoms, Φ, ρ, ρdot, σ, σdot)
    # Intrinsic framework acceleration
    ddρ = -4 * π^2 * Ξ * ρ

    # Get the indices of the coordinates of each atom in 'atoms' interacting with the mobile particle
    atom_indices = [
        [
            get(index, c, ErrorException("Coordinate not found")) for
            c in [(a..., d) for d in [1, 2, 3]]
        ] for a in atoms
    ]

    # Get the equilibrium positions of each atom in 'atoms'. Atom (1,1,1) is at the origin
    atom_positions = [ρ[atom_indices[a]] .+ (atoms[a] .- 1) .* α for a in eachindex(atoms)]

    # Calculate the negative gradient of the interaction with each atom in 'atoms' and multiply its
    # by 4π² to get the force

    F_ρ = [-4 * π^2 .* ForwardDiff.gradient(Φ, pos - σ) for pos in atom_positions]
    # Force on mobile particles due to interaction using Newton's 3d law
    F_σ = -sum(F_ρ)

    # Add the force to ddρ to get the full acceleration (atom mass = 1, so no mass division)
    view(ddρ, vcat(atom_indices...)) .+= vcat(F_ρ...)

    return (ρdot, ddρ, σdot, F_σ ./ μ)
end

# 5th order Runge-Kutta step. 
# current_state is (ρ, ρdot, σ, σdot)
# sys is (index, Ξ)
function RKstep(μ, α, sys, atoms, Φ, current_state, δτ)
    k1 = derivatives(μ, α, sys..., atoms, Φ, current_state...)
    k2 = derivatives(μ, α, sys..., atoms, Φ, (current_state .+ k1 .* (δτ / 4))...)
    k3 = derivatives(μ, α, sys..., atoms, Φ, (current_state .+ (k1 .+ k2) .* (δτ / 8))...)
    k4 = derivatives(
        μ,
        α,
        sys...,
        atoms,
        Φ,
        (current_state .+ k3 .* δτ .- k2 .* (δτ / 2))...,
    )
    k5 = derivatives(
        μ,
        α,
        sys...,
        atoms,
        Φ,
        (current_state .+ k1 .* (δτ * 3 / 16) .+ k4 .* (δτ * 9 / 16))...,
    )
    k6 = derivatives(
        μ,
        α,
        sys...,
        atoms,
        Φ,
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

function dynamical_matrix(coupling::Vector{Coupling})
    res = function f(q)
        D = zeros(ComplexF64, 3, 3)
        for c in coupling
            D[c.d1, c.d2] += c.k * exp(-1im * dot(q, c.disp))
        end
        return D
    end

    return res

end

function dynamical_matrix_small(coupling::Vector{Coupling})
    res = function f(θ, ϕ)
        D = zeros(ComplexF64, 3, 3)
        for c in coupling
            D[c.d1, c.d2] +=
                c.k * (-1im * dot([cos(ϕ) * sin(θ), sin(ϕ) * sin(θ), cos(θ)], c.disp))^2 / 2
        end
        return D
    end

    return res

end


# Single-pass loss for a 1D system with Gaussian interaction
function Δ_1D_analytic(σ_dot, Φ0, λ, s, ω_par, ω_perp, ν)
    Y = 2 * π * λ / σ_dot

    Δ_transverse =
        (π * Y^2 * s^2 / λ^4 * Φ0^2) *
        exp(-s^2 / λ^2) *
        exp(-Y^2 * (ν^2 + 2 * ω_perp^2)) *
        besseli(0, 2 * ω_perp^2 * Y^2)

    Δ_long =
        (π * Y^4 / λ^2 * Φ0^2) *
        exp(-s^2 / λ^2) *
        exp(-Y^2 * (ν^2 + 2 * ω_par^2)) *
        (
            (ν^2 + 2 * ω_par^2) * besseli(0, 2 * ω_par^2 * Y^2) -
            2 * ω_par^2 * besseli(1, 2 * ω_par^2 * Y^2)
        )

    return (Δ_transverse, Δ_long)
end

# Calculate the loss for a single pass when the particle moves with a constant speed
# in the z-direction with impact parameter s and the width/strength of the Gaussian
# interaction is given by λ/Φ0
function numerical_Δ(Φ0, λ, s, σ_dot, DynamicalMatrix)
    function fun_int(q)
        # Solve the eigenproblem
        eig = eigen(DynamicalMatrix(q))
        ωs = sqrt.(eig.values)
        ηs = eig.vectors
        # Displacement vector
        σ0 = [0, s, 0]

        potential_gradient = [
            √(2 * π) *
            λ *
            (σ0 ./ λ^2 + 2im * π * [0, 0, ω] ./ σ_dot) *
            exp((-dot(σ0, σ0) - 4 * π^2 * λ^4 * ω^2 / σ_dot^2) / 2 / λ^2) for ω in ωs
        ]
        mode_coupling = [abs(dot(ηs[:, jj], potential_gradient[jj]) / σ_dot) for jj = 1:3]
        res = dot(mode_coupling, mode_coupling) * 2 * π^2 * Φ0^2
        return res
    end
    res = hcubature(
        q -> fun_int(q) / (2 * π)^3,
        0.0 .* ones(3),
        2.0 .* π .* ones(3),
        rtol = 1e-3,
        initdiv = 30,
    )
    return res
end

function Loss_Matrix(DynamicalMatrix_Small)
    function fun_int(q)
        # Solve the eigenproblem
        eig = eigen(DynamicalMatrix_Small(q...))
        ωs = sqrt.(eig.values)
        ηs = eig.vectors
        res = sum([(ηs[:, jj] * ηs[:, jj]') ./ ωs[jj]^3 for jj = 1:3])
        return res
    end
    res = hcubature(
        q -> sin(q[1]) .* fun_int(q) / 32 / π^3,
        [0, 0],
        [π, 2 * π],
        rtol = 1e-4,
        # initdiv = 20,
    )
    return res
end
