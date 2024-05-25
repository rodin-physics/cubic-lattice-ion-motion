include("../main.jl")
include("../plotting.jl")

## PARAMETERS
μ = [Inf, Inf, 1]           # Mass of the particle in the units of chain masses
α = 2                       # Lattice constant
k1 = 15                     # Nearest neighbor force constant
k2 = 5                      # Next-nearest neighbor force constant
δτ = 1e-3                   # Time step

τ_max = 30
nPts = floor(τ_max / δτ) |> Int

init_pos = 3.5 * α
# init_speed = 2
init_speed = 7.5
# σ = [α / 2, α / 2, init_pos]
σ = [0, 0, init_pos]
σ_dot = [0, 0, init_speed]


# atoms = [(x, y, z) for x = 1:2, y = 1:2, z = 1:8] |> vec
atoms =
    [(1, 1, 1), (1, 1, 2), (1, 1, 3), (1, 1, 4), (1, 1, 5), (1, 1, 6), (1, 1, 7), (1, 1, 8)]

size_x = 100
size_y = 100
size_z = 100

Φ0 = 1
λ = 1 / 2
@inline function Φ(r)
    res = Φ0 * exp(-dot(r, r) / 2 / λ^2)
    return res
end

function Φ_total(σ)
    res = [Φ(σ - α .* [l .- (1, 1, 1)...]) for l in atoms] |> sum
    return res
end
# Φ_total([α/2, α/2,  3.5 * α])
# (Φ_total([α/2, α/2,  4. * α])-Φ_total([α/2, α/2,  3.5 * α])) *8 * pi^2 |> sqrt


function Φ_total_grad(σ)
    res = vcat(
        [ForwardDiff.gradient(x -> Φ(x - α .* [l .- (1, 1, 1)...]), σ) for l in atoms]...,
    )
    return res
end

# Dynamical matrix
function DynamicalMatrix(qx, qy, qz)
    nearest = 2 .* k1 .* [
        (1-cos(qx)) 0 0
        0 (1-cos(qy)) 0
        0 0 (1-cos(qz))
    ]

    next_nearest =
        2 .* k2 .* [
            2-cos(qx)*cos(qy)-cos(qx)*cos(qz) sin(qx)*sin(qy) sin(qx)*sin(qz)
            sin(qx)*sin(qy) 2-cos(qx)*cos(qy)-cos(qy)*cos(qz) sin(qz)*sin(qy)
            sin(qx)*sin(qz) sin(qz)*sin(qy) 2-cos(qz)*cos(qy)-cos(qx)*cos(qz)
        ]

    return (nearest + next_nearest)
end

function DynamicalMatrix_Small(θ, ϕ)
    res = ForwardDiff.derivative(
        k -> ForwardDiff.derivative(
            q -> DynamicalMatrix(q * cos(ϕ) * sin(θ), q * sin(ϕ) * sin(θ), q * cos(θ)),
            k,
        ),
        0,
    )
    return (res ./ 2)
end

function polaron(L)
    function fun_int(q)
        # Solve the eigenproblem
        eig = eigen(DynamicalMatrix(q...))
        ωs2 = eig.values
        ηs = eig.vectors

        res = sum([(ηs[:, jj] * ηs[:, jj]') ./ ωs2[jj] for jj = 1:3])
        return res
    end
    res = hcubature(
        q -> real.(fun_int(q) * exp(1im * dot(q, L)) / (2 * π)^3),
        0.0 .* ones(3),
        2.0 .* π .* ones(3),
        rtol = 1e-3,
        # initdiv = 30,
    )

end
# @time polaron(atoms[32])[1]
polarons = [polaron([atoms[1] .- a...])[1] for a in atoms]
polaron_matrix = [polarons[abs.(j - k)+1] for j in eachindex(atoms), k in eachindex(atoms)]
polaron_matrix = vcat([hcat(polaron_matrix[r, :]...) for r in eachindex(atoms)]...)

# Set up the lattice
coords = ["x", "y", "z"]

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
        [
            Coupling(v, coords[n], coords[m], -k1 * v[n] * v[m] / dot(v, v)) for
            n = 1:3, m = 1:3
        ] |> vec for v in N_disp
    ]...,
)
N_couplings = filter(x -> x.k != 0, N_couplings)

NN_couplings = vcat(
    [
        [
            Coupling(v, coords[n], coords[m], -k2 * v[n] * v[m] / dot(v, v)) for
            n = 1:3, m = 1:3
        ] |> vec for v in NN_disp
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
    [
        Coupling((0, 0, 0), coords[n], coords[m], self_coupling_matrix[n, m]) for n = 1:3,
        m = 1:3
    ] |> vec
self_coupling = filter(x -> x.k != 0, self_coupling)
couplings = vcat(N_couplings, NN_couplings, self_coupling)

# Make the system
sys = system(size_x, size_y, size_z, couplings)
println("Starting calculations...")

## SIMULATIONS
# Compute the particle trajectory using the full system

pos = Vector{Vector{Float64}}(undef, nPts)
speed = Vector{Vector{Float64}}(undef, nPts)
pos[1] = σ
speed[1] = σ_dot
ρ_init = zeros(length(sys[1]))
ρ_dot_init = zeros(length(sys[1]))
function solve_full()
    current_state = (ρ_init, ρ_dot_init, σ, σ_dot)

    @showprogress for ii = 2:nPts
        current_state = RKstep(μ, α, sys, atoms, Φ, current_state, δτ)
        pos[ii] = current_state[3]
        speed[ii] = current_state[4]
    end
    return (pos, speed)
end
if !isfile("Data/3D/Local_Loss.jld2")
    pos, speed = solve_full()
end

M_matrix = (Loss_Matrix(DynamicalMatrix_Small)[1])[3, 3]

function polaron_potential(σ)
    res = Φ_total_grad(σ)' * polaron_matrix * Φ_total_grad(σ)
    return res
end

function total_potential(σ)
    res = Φ_total(σ) - polaron_potential(σ)
    return res
end
# total_potential(σ)
# total_potential([0,0,3α])-total_potential([0,0,3.5α])
function derivative_particle(σ, σdot)
    F_potential = -ForwardDiff.gradient(total_potential, σ)

    Hess = ForwardDiff.derivative(
        y -> ForwardDiff.derivative(x -> Φ_total([σ[1], σ[2], x]), y),
        σ[3],
    )
    friction = -Hess^2 * M_matrix * σdot

    res = (σdot, 4 * π^2 * (F_potential + friction) ./ μ)
    return res
end

function RKstep_particle(current_state)
    k1 = derivative_particle(current_state...)
    k2 = derivative_particle((current_state .+ k1 .* (δτ / 4))...)
    k3 = derivative_particle((current_state .+ (k1 .+ k2) .* (δτ / 8))...)
    k4 = derivative_particle((current_state .+ k3 .* δτ .- k2 .* (δτ / 2))...)
    k5 = derivative_particle(
        (current_state .+ k1 .* (δτ * 3 / 16) .+ k4 .* (δτ * 9 / 16))...,
    )
    k6 = derivative_particle(
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

pos_particle = Vector{Vector{Float64}}(undef, nPts)
speed_particle = Vector{Vector{Float64}}(undef, nPts)
pos_particle[1] = σ
speed_particle[1] = σ_dot
function solve_particle()
    current_state = (σ, σ_dot)
    @showprogress for ii = 2:nPts
        current_state = RKstep_particle(current_state)
        pos_particle[ii] = current_state[1]
        speed_particle[ii] = current_state[2]
    end
    return (pos_particle, speed_particle)

end
if !isfile("Data/3D/Local_Loss.jld2")
    pos_particle, speed_particle = solve_particle()
end

if !isfile("Data/3D/Local_Loss.jld2")
    save_object(
        "Data/3D/Local_Loss.jld2",
        (pos, speed, pos_particle, speed_particle, δτ, Φ0, λ),
    )
end

d = load_object("Data/3D/Local_Loss.jld2")
pos = d[1]
speed = d[2]
pos_particle = d[3]
speed_particle = d[4]

## FIGURES
set_theme!(CF_theme)
colors = [CF_vermillion, CF_orange, CF_green, CF_sky]
fig = Figure(size = (1200, 1000))

supertitle = fig[1, 1]
Label(
    supertitle,
    "Trapped dissipation",
    tellwidth = false,
    tellheight = false,
    font = :latex,
    fontsize = 36,
    valign = :center,
)

main_grid = fig[2:30, 1] = GridLayout()
legend_grid = fig[31, 1] = GridLayout()

ax_pos = Axis(
    main_grid[1, 1],
    # title = "Position",
    ylabel = L"Position $\sigma$",
    xlabel = L"Time $\tau$",
)

ax_speed = Axis(
    main_grid[2, 1],
    # title = "Velocity",
    ylabel = L"Velocity $\dot{\sigma}$",
    xlabel = L"Time $\tau$",
)
ax = [ax_pos, ax_speed]
labs = ["(a)", "(b)"]

for ii = 1:2
    text!(
        ax[ii],
        0.95,
        0.95,
        text = labs[ii],
        align = (:right, :top),
        space = :relative,
        fontsize = 36,
        font = :latex,
        color = :black,
    )
end

lines!(ax_pos, δτ .* (1:length(pos)), [x[3] for x in pos], color = CF_sky, linewidth = 2)
lines!(
    ax_speed,
    δτ .* (1:length(speed)),
    [x[3] for x in speed],
    color = CF_sky,
    linewidth = 2,
)

lines!(
    ax_pos,
    δτ .* (1:length(pos_particle)),
    [x[3] for x in pos_particle],
    color = CF_vermillion,
    linewidth = 2,
)
lines!(
    ax_speed,
    δτ .* (1:length(pos_particle)),
    [x[3] for x in speed_particle],
    color = CF_vermillion,
    linewidth = 2,
)
hidexdecorations!(ax_pos)
polys =
    [PolyElement(color = c, strokecolor = :transparent) for c in [CF_sky, CF_vermillion]]

Legend(
    legend_grid[1, 1],
    [polys],
    [["Full solution", "Time-local solution"]],
    [""],
    halign = :center,
    valign = :center,
    tellheight = false,
    tellwidth = false,
    framevisible = false,
    orientation = :horizontal,
    titlevisible = false,
    titleposition = :left,
    titlefont = :latex,
)

fig
save("Local_time.pdf", fig)
