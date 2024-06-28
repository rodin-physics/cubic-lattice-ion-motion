include("system.jl")

function derivatives(m, M, a, index, V, atoms, U, r, rdot, R, Rdot)
    # Intrinsic framework acceleration
    F_r = -V * r

    # Get the indices of the coordinates of each atom in 'atoms' interacting with the mobile particle
    atom_indices = [
        [
            get(index, c, ErrorException("Coordinate not found")) for
            c in [(a..., d) for d in [1, 2, 3]]
        ] for a in atoms
    ]

    # Get the equilibrium positions of each atom in 'atoms'. Atom (1,1,1) is at the origin
    atom_positions = [r[atom_indices[n]] .+ (atoms[n] .- 1) .* a for n in eachindex(atoms)]

    # Calculate the negative gradient of the interaction with each atom in 'atoms'

    F_r_particle = [-1 .* ForwardDiff.gradient(U, pos - R) for pos in atom_positions]
    # Force on mobile particles due to interaction using Newton's 3d law
    F_R = -sum(F_r)

    # Add the force to ddρ to get the full acceleration (atom mass = 1, so no mass division)
    view(F_r, vcat(atom_indices...)) .+= vcat(F_r_particle...)

    return (rdot, F_r ./ m, Rdot, F_R ./ M)
end

# 5th order Runge-Kutta step. 
# current_state is (r, rdot, R, Rdot)
# sys is (index, V)
function RKstep(m, M, a, sys, atoms, U, current_state, δt)
    k1 = derivatives(m, M, a, sys..., atoms, U, current_state...)
    k2 = derivatives(m, M, a, sys..., atoms, U, (current_state .+ k1 .* (δt / 4))...)
    k3 =
        derivatives(m, M, a, sys..., atoms, U, (current_state .+ (k1 .+ k2) .* (δt / 8))...)
    k4 = derivatives(
        m,
        M,
        a,
        sys...,
        atoms,
        U,
        (current_state .+ k3 .* δt .- k2 .* (δt / 2))...,
    )
    k5 = derivatives(
        m,
        M,
        a,
        sys...,
        atoms,
        U,
        (current_state .+ k1 .* (δt * 3 / 16) .+ k4 .* (δt * 9 / 16))...,
    )
    k6 = derivatives(
        m,
        M,
        a,
        sys...,
        atoms,
        U,
        (
            current_state .- k1 .* (3 / 7 * δt) .+ k2 .* (2 / 7 * δt) .+
            k3 .* (12 / 7 * δt) .- k4 .* (12 / 7 * δt) .+ k5 .* (8 / 7 * δt)
        )...,
    )
    res =
        current_state .+
        (δt / 90) .* (7 .* k1 .+ 32 .* k3 .+ 12 .* k4 .+ 32 .* k5 .+ 7 .* k6)
    return res
end

# q here is q * a
function dynamical_matrix(m, coupling::Vector{Coupling})
    res = function f(q)
        D = zeros(ComplexF64, 3, 3)
        for c in coupling
            D[c.d1, c.d2] += c.k * exp(-1im * dot(q, c.disp)) ./ m
        end
        return D
    end

    return res

end

# Expanded dynamical matrix to q². The matrix has been divided by q².
function dynamical_matrix_small(a, m, coupling::Vector{Coupling})
    res = function f(θ, ϕ)
        D = zeros(ComplexF64, 3, 3)
        for c in coupling
            D[c.d1, c.d2] +=
                a^2 .* c.k *
                (-1im * dot([cos(ϕ) * sin(θ), sin(ϕ) * sin(θ), cos(θ)], c.disp))^2 /
                2 ./ m
        end
        return D
    end

    return res

end

function Loss_Matrix(a, m, DynamicalMatrix_Small)
    ρ = m / a^3
    function fun_int(q)
        # Solve the eigenproblem
        eig = eigen(DynamicalMatrix_Small(q...))
        Ωs = sqrt.(eig.values)
        ηs = eig.vectors
        res = sum([(ηs[:, jj] * ηs[:, jj]') ./ Ωs[jj]^3 for jj = 1:3])
        return res
    end
    res = hcubature(
        q -> sin(q[1]) .* fun_int(q) / 16 / π^2 / ρ,
        [0, 0],
        [π, 2 * π],
        rtol = 1e-4,
        # initdiv = 20,
    )
    return res
end


function Corr(ħ, m, dyn_mat, ΩT, t, disp)
    function fun_int(q)
        # Solve the eigenproblem
        eig = eigen(dyn_mat(q))
        Ωs = sqrt.(eig.values)
        ηs = eig.vectors

        res =
            [
                (ηs[:, ii] * ηs[:, ii]') / 2 / Ωs[ii] * coth(Ωs[ii] / 2 / ΩT) .*
                cos(Ωs[ii] * t) for ii = 1:3
            ] |> sum
        res = res .* exp(1im * dot(q, disp))
        return real(res)
    end
    res =
        hcubature(
            q -> fun_int(q)[1, 1] / (2 * π)^3,
            0.0 .* ones(3),
            2.0 .* π .* ones(3),
            rtol = 1e-2,
            # initdiv = 20,
            atol = 1e-6,
        ) ./ m .* ħ
    return res
end

function Recoil(m, dyn_mat, t, disp)
    function fun_int(q)
        # Solve the eigenproblem
        eig = eigen(dyn_mat(q))
        Ωs = sqrt.(eig.values)
        ηs = eig.vectors

        res = [(ηs[:, ii] * ηs[:, ii]') / Ωs[ii]^2 .* cos(Ωs[ii] * t) for ii = 1:3] |> sum
        res = res .* exp(1im * dot(q, disp))
        return real(res)[1, 1]
    end
    res =
        hcubature(
            q -> fun_int(q) / (2 * π)^3,
            0.0 .* ones(3),
            2.0 .* π .* ones(3),
            rtol = 1e-2,
            initdiv = 20,
            # atol = 1e-6,
        ) ./ m
    return res
end

# ## Generating Homogeneous motion

# # Mode amplitude
# function ζq(ωq, ωT)
#     η = 1e-12
#     # Subtract a small number from p. The reason is that for low ωT, p ≈ 1,
#     # causing issues with the rand() generator
#     n = rand(Geometric(1 - exp(-ωq / ωT) - η))
#     res = √(n + 1 / 2) * √(2 / ωq)
#     return res
# end

# function homogeneous_motion(dyn_mat, ωT)
#     eig = eigen(dyn_mat)
#     ωs = sqrt.(eig.values) |> real
#     ζs = [ζq(ωq, ωT) for ωq in ωs]
#     ηs = eig.vectors
#     ϕs = 2 * π * rand(length(ωs))
#     disp = [ζs[ii] .* ηs[:, ii] * exp(1im * ϕs[ii]) for ii = 1:3] |> sum
#     speed =
#         [-2im * π .* ωs[ii] .* ζs[ii] .* ηs[:, ii] * exp(1im * ϕs[ii]) for ii = 1:3] |> sum
#     return disp, speed

# end


function relaxation(sys, R, λ, U0, nStep)
    m = 1
    M = 1
    atoms = Iterators.product([1:size_x, 1:size_y, 1:size_z]...) |> collect |> vec
    atom_indices = [
        [
            get(sys[1], c, ErrorException("Coordinate not found")) for
            c in [(a..., d) for d in [1, 2, 3]]
        ] for a in atoms
    ]
    @inline function U(r)
        res = U0 * exp(-dot(r, r) / 2 / λ^2)
        return res
    end

    r = zeros(length(sys[1]))
    r_dot = zeros(length(sys[1]))
    if nStep > 0

        @showprogress for _ = 1:nStep
            d = derivatives(m, M, a, sys..., atoms, U, r, r_dot, R, [0, 0, 0])
            step = min(1e-1, maximum(abs.(d[2])))
            r += step .* normalize(d[2])
        end

        @showprogress for _ = 1:nStep
            d = derivatives(m, M, a, sys..., atoms, U, r, r_dot, R, [0, 0, 0])
            step = min(1e-2, maximum(abs.(d[2])))
            r += step .* normalize(d[2])
        end

        @showprogress for _ = 1:nStep
            d = derivatives(m, M, a, sys..., atoms, U, r, r_dot, R, [0, 0, 0])
            step = min(1e-3, maximum(abs.(d[2])))
            r += step .* normalize(d[2])
        end

        @showprogress for _ = 1:nStep
            d = derivatives(m, M, a, sys..., atoms, U, r, r_dot, R, [0, 0, 0])
            step = min(1e-4, maximum(abs.(d[2])))
            r += step .* normalize(d[2])
        end

        @showprogress for _ = 1:nStep
            d = derivatives(m, M, a, sys..., atoms, U, r, r_dot, R, [0, 0, 0])
            step = min(1e-5, maximum(abs.(d[2])))
            r += step .* normalize(d[2])
        end

    end
    atom_positions = [r[atom_indices[n]] .+ (atoms[n] .- 1) .* a for n in eachindex(atoms)]
    interaction = [U(pos - R) for pos in atom_positions] |> sum
    elastic = r' * sys[2] * r / 2
    max_disp = maximum(abs.(r))
    d = derivatives(m, M, a, sys..., atoms, U, r, r_dot, R, [0, 0, 0])
    return (interaction, elastic, max_disp, d[2], r)
end
