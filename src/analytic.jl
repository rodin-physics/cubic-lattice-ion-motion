using HCubature
using SpecialFunctions

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
