include("../src/unitful.jl")
include("../src/plotting.jl")

dirs = ["Data", "Data/3D_Finite_T"]
[isdir(d) ? nothing : mkdir(d) for d in dirs]

## PARAMETERS
a = 3                       # Lattice constant in Å
k1 = 520                    # Nearest neighbor force constant in meV / Å²
k2 = 170                    # Next-nearest neighbor force constant in meV / Å²
m = 3.5                     # Lattice mass in meV * (ps / Å)²
ħ = 0.6582119569            # Planck constant in meV * ps
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

# Dynamical matrix
DynamicalMatrix = dynamical_matrix(m, couplings)

# BAND STRUCTURE
Γ = [0, 0, 0]
X = [π, 0, 0]
M = [π, π, 0]
R = [π, π, π]

nPts = 200

ΓX = [Γ + (X - Γ) ./ nPts * n for n = 0:nPts]
XM = [X + (M - X) ./ nPts * n for n = 1:nPts]
MΓ = [M + (Γ - M) ./ nPts * n for n = 1:nPts]
ΓR = [Γ + (R - Γ) ./ nPts * n for n = 1:nPts]

path = vcat([ΓX, XM, MΓ, ΓR]...)
energies = [sqrt.(real.(eigen(DynamicalMatrix(v)).values)) for v in path]

plot_positions = zeros(length(path))

for ii in eachindex(path)[2:end]
    plot_positions[ii] = plot_positions[ii-1] + norm(path[ii] - path[ii-1])
end

## RECOIL
if !isfile("Data/3D_Finite_T/Recoil.jld2")
    nPts = 200
    res = zeros(nPts)
    tmin = 0
    tmax = 2
    ts = range(tmin, tmax, length = nPts)
    pr = Progress(nPts)
    Threads.@threads for ii in eachindex(res)
        res[ii] = Recoil(m, DynamicalMatrix, ts[ii], [0, 0, 0])[1]
        next!(pr)
        GC.safepoint()
    end
    save_object("Data/3D_Finite_T/Recoil.jld2", (ts, res))
end

## FIGURES

set_theme!(CF_theme)
colors = [CF_vermillion, CF_orange, CF_green, CF_sky]

fig = Figure(size = (1200, 400))

ax_bands = Axis(
    fig[1, 1],
    title = "Phonon dispersion",
    ylabel = "Energy (meV)",
    # xlabel = L"Initial speed $\dot{\sigma}_0$",
)

ax_recoil = Axis(
    fig[1, 2],
    title = "Recoil kernel",
    ylabel = L"C_{11}(t) / C_{11}(0)",
    xlabel = L"$t$ (ps)",
)

for ii = 1:3
    lines!(
        ax_bands,
        plot_positions,
        [e[ii] for e in energies] .* ħ,
        linewidth = 4,
        color = CF_vermillion,
    )
end

x_ticks = [
    plot_positions[1],
    plot_positions[nPts+1],
    plot_positions[2*nPts+1],
    plot_positions[3*nPts+1],
    plot_positions[4*nPts+1],
]
x_labels = [L"Γ", "X", "M", L"\Gamma", "R"]

ax_bands.xticks = (x_ticks, x_labels)
xlims!(ax_bands, (plot_positions[1], plot_positions[end]))
ylims!(ax_bands, (0, 22))

(ts, res) = load_object("Data/3D_Finite_T/Recoil.jld2")
lines!(ax_recoil, ts, res ./ res[1], linewidth = 4, color = CF_sky)

fig
save("Phonon_dispersion.pdf", fig)
# sum(res) .* (ts[2]-ts[1])

# r = [Recoil(m, DynamicalMatrix, 0, [n, 0, 0])[1] for n in 0 : 55]
# scatter(log.(1:56), log.(abs.(r)./ r[1]))

# r

# DS =  dynamical_matrix_small(a, m, couplings)

# Loss_Matrix(a, m, DS)
