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
## Generating Homogeneous motion

# Mode amplitude
function ζq(ωq, ωT)
    η = 1e-12
    # Subtract a small number from p. The reason is that for low ωT, p ≈ 1,
    # causing issues with the rand() generator
    n = rand(Geometric(1 - exp(-ωq / ωT) - η))
    res = √(n + 1 / 2) * √(2 / ωq)
    return res
end

function homogeneous_motion(dyn_mat, ωT)
    eig = eigen(dyn_mat)
    ωs = sqrt.(eig.values) |> real
    ζs = [ζq(ωq, ωT) for ωq in ωs]
    ηs = eig.vectors
    ϕs = 2 * π * rand(length(ωs))
    disp = [ζs[ii] .* ηs[:, ii] * exp(1im * ϕs[ii]) for ii = 1:3] |> sum
    speed =
        [-2im * π .* ωs[ii] .* ζs[ii] .* ηs[:, ii] * exp(1im * ϕs[ii]) for ii = 1:3] |> sum
    return disp, speed

end

function Corr(dyn_mat, ωT, τ, disp)
    function fun_int(q)
        # Solve the eigenproblem
        eig = eigen(dyn_mat(q))
        ωs = sqrt.(eig.values)
        ηs = eig.vectors

        res =
            [
                (ηs[:, ii] * ηs[:, ii]') / 2 / ωs[ii] * coth(ωs[ii] / 2 / ωT) .*
                exp(-2im * π * ωs[ii] * τ) for ii = 1:3
            ] |> sum
        res = res .* exp(1im * dot(q, disp))
        return real(res)
    end
    res = hcubature(
        q -> fun_int(q) / (2 * π)^3,
        0.0 .* ones(3),
        2.0 .* π .* ones(3),
        rtol = 1e-2,
        initdiv = 30,
    )
    return res
end

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

        @showprogress for _ = 1:nStep
            d = derivatives(μ, α, sys..., atoms, Φ, ρ, ρ_dot, σ, [0, 0, 0])
            step = min(1e-1, maximum(abs.(d[2])))
            ρ += step .* normalize(d[2])
        end

        @showprogress for _ = 1:nStep
            d = derivatives(μ, α, sys..., atoms, Φ, ρ, ρ_dot, σ, [0, 0, 0])
            step = min(1e-2, maximum(abs.(d[2])))
            ρ += step .* normalize(d[2])
        end

        @showprogress for _ = 1:nStep
            d = derivatives(μ, α, sys..., atoms, Φ, ρ, ρ_dot, σ, [0, 0, 0])
            step = min(1e-3, maximum(abs.(d[2])))
            ρ += step .* normalize(d[2])
        end

        @showprogress for _ = 1:nStep
            d = derivatives(μ, α, sys..., atoms, Φ, ρ, ρ_dot, σ, [0, 0, 0])
            step = min(1e-4, maximum(abs.(d[2])))
            ρ += step .* normalize(d[2])
        end

        @showprogress for _ = 1:nStep
            d = derivatives(μ, α, sys..., atoms, Φ, ρ, ρ_dot, σ, [0, 0, 0])
            step = min(1e-5, maximum(abs.(d[2])))
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
