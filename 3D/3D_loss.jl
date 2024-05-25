include("../main.jl")
include("../plotting.jl")

## PARAMETERS
μ = [Inf, Inf, 1]           # Mass of the particle in the units of chain masses
α = 2                       # Lattice constant
k1 = 15                     # Nearest neighbor force constant
k2 = 5                      # Next-nearest neighbor force constant
δτ = 1e-3                   # Time step

size_x = 50
size_y = 50
size_z = 50

Φ0 = 0.01
s_vals = [0, 1]
λ_vals = [1 / 2, 1, 2, 4]
params = [(s, λ) for s in s_vals, λ in λ_vals] |> vec

nPts = 50
speed_min = 1
speed_max = 80
init_speeds = range(speed_min, speed_max, length = nPts)

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
atoms = [(1, 1, 1)]           # The moving particle interacts with a single atom

function Φ_full(σ, λ, atoms, α)
    res = [exp(-norm(σ .- (a .- (1, 1, 1)) .* α)^2 / 2 / λ^2) for a in atoms] |> sum
    return res
end

function Hess_Φ(σ, λ, atoms, α)
    res = ForwardDiff.hessian(σ -> Φ_full(σ, λ, atoms, α), σ)
    return res
end

M_loss = Loss_Matrix(DynamicalMatrix_Small)[1]

# Δ/(σ̇ Φ0²) in low-speed limit
function Δ_pass(σ0, σ_dot, λ, atoms, α)
    σ_dot_unit = normalize(σ_dot)
    r = quadgk(
        t ->
            σ_dot_unit' *
            Hess_Φ(σ_dot_unit * t + σ0, λ, atoms, α) *
            M_loss *
            Hess_Φ(σ_dot_unit * t + σ0, λ, atoms, α) *
            σ_dot_unit,
        -Inf,
        Inf,
    )
    return r[1]

end

println("Starting calculations...")

## SIMULATIONS
for p in params
    s = p[1]
    λ = p[2]

    if !isfile("Data/3D/Loss_s$(s)_λ$(λ).jld2")

        # Interaction function
        @inline function Φ(r)
            res = Φ0 * exp(-dot(r, r) / 2 / λ^2)
            return res
        end

        loss = zeros(nPts)
        numerical_loss = zeros(nPts)
        prog = Progress(nPts)
        Threads.@threads for ii in eachindex(loss)
            init_pos = -7 * λ
            init_speed = init_speeds[ii]
            σ = [0, s, init_pos]
            σ_dot = [0, 0, init_speed]

            ρ_init = zeros(length(sys[1]))
            ρ_dot_init = zeros(length(sys[1]))

            current_state = (ρ_init, ρ_dot_init, σ, σ_dot)
            while current_state[3][3] < 7 * λ
                current_state = RKstep(μ, α, sys, atoms, Φ, current_state, δτ)
            end
            final_speed = current_state[4][3]
            loss[ii] = (init_speed^2) / (8 * π^2) - (final_speed^2) / (8 * π^2)
            numerical_loss[ii] = numerical_Δ(Φ0, λ, s, init_speed, DynamicalMatrix, 3)[1]
            next!(prog)
        end

        save_object(
            "Data/3D/Loss_s$(s)_λ$(λ).jld2",
            (s, λ, init_speeds, loss, numerical_loss),
        )
    end
end

## FIGURES
set_theme!(CF_theme)
colors = [CF_vermillion, CF_orange, CF_green, CF_sky]
fig = Figure(size = (1200, 1600))

supertitle = fig[1, 1]
Label(
    supertitle,
    "Loss in 3D Lattice",
    tellwidth = false,
    tellheight = false,
    font = :latex,
    fontsize = 36,
    valign = :center,
)

main_grid = fig[2:30, 1] = GridLayout()
legend_grid = fig[31, 1] = GridLayout()

head_on_data =
    load_object.([
        "Data/3D/Loss_s0_λ0.5.jld2",
        "Data/3D/Loss_s0_λ1.0.jld2",
        "Data/3D/Loss_s0_λ2.0.jld2",
        "Data/3D/Loss_s0_λ4.0.jld2",
    ])

offset_data =
    load_object.([
        "Data/3D/Loss_s1_λ0.5.jld2",
        "Data/3D/Loss_s1_λ1.0.jld2",
        "Data/3D/Loss_s1_λ2.0.jld2",
        "Data/3D/Loss_s1_λ4.0.jld2",
    ])

ax_head_on = Axis(
    main_grid[1, 1],
    yscale = log10,
    title = "Head-on",
    ylabel = L"$\Delta/\dot{\sigma}_0\Phi_0^2$",
    xlabel = L"Initial speed $\dot{\sigma}_0$",
)
ax_offset = Axis(
    main_grid[2, 1],
    title = "Offset",
    yscale = log10,
    ylabel = L"$\Delta/\dot{\sigma}_0\Phi_0^2$",
    xlabel = L"Initial speed $\dot{\sigma}_0$",
)
ax = [ax_head_on, ax_offset]
data = [head_on_data, offset_data]
labs = ["(a)", "(b)"]

for jj in eachindex(ax)
    text!(
        ax[jj],
        0.95,
        0.95,
        text = labs[jj],
        align = (:right, :top),
        space = :relative,
        fontsize = 36,
        font = :latex,
        color = :black,
    )
    for ii in eachindex(data[jj])
        d = data[jj][ii]
        scatter!(
            ax[jj],
            d[3][2:end],
            (d[5]./d[3])[2:end] ./ Φ0^2,
            color = colors[ii],
            marker = '□',
            markersize = 20,
        )

        scatter!(
            ax[jj],
            d[3][2:end],
            (d[4]./d[3])[2:end] ./ Φ0^2,
            color = colors[ii],
            marker = :cross,
            markersize = 20,
        )

        s = d[1]
        λ = d[2]
        c = Δ_pass([0, s, 0], [0, 0, 1], λ, atoms, α)
        # c =
        #     exp(-s^2 / λ^2) * √(π) * (2 * s^2 + 3 * λ^2) / 4 / λ^5 *
        #     (Loss_Matrix(DynamicalMatrix_Small)[1])[3, 3]

        lines!(ax[jj], d[3][2:end], c * ones(nPts - 1), color = colors[ii], linewidth = 2)
    end
end

hidexdecorations!(ax_head_on)

polys = [PolyElement(color = c, strokecolor = :transparent) for c in colors]
sources = [
    MarkerElement(
        marker = '□',
        color = :black,
        strokecolor = :transparent,
        markersize = 20,
    ),
    MarkerElement(
        marker = :cross,
        color = :black,
        strokecolor = :transparent,
        markersize = 20,
    ),
]

Legend(
    legend_grid[1, 1],
    [polys, sources],
    [["1/2", "1", "2", "4"], ["Numeric", "Simulation", "Analytic"]],
    [L"$\lambda$:", " "],
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

save("3D_Loss.pdf", fig)
