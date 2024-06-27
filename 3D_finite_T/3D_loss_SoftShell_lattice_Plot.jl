include("3D_loss_SoftShell_lattice.jl")
include("../src/plotting.jl")


atom_sets = [face_atoms, cube_atoms]
cs = [zeros(length(λs)), zeros(length(λs))]

for ii in eachindex(λs)
    λ = λs[ii]
    σ0 = [α / 2, α / 2, 0]
    σ_dot = [0, 0, 1]

    @inline function Φ(r)
        res = Φ0 / (1 + exp((norm(r) - 3) / λ))
        return (isnan(res) ? 0 : res)
    end

    function Φ_full(σ, atoms)
        res = [Φ((a .- (1, 1, 1)) .* α .- σ) for a in atoms] |> sum
        return res
    end

    function Hess_Φ(σ, atoms)
        res = ForwardDiff.hessian(σ -> Φ_full(σ, atoms), σ)
        return res
    end
    M_loss = Loss_Matrix(DynamicalMatrixSmall)[1]

    # Δ/(σ̇ Φ0²) in low-speed limit
    function Δ_pass(σ0, σ_dot, atoms)
        σ_dot_unit = normalize(σ_dot)
        r = quadgk(
            t ->
                σ_dot_unit' *
                Hess_Φ(σ_dot_unit * t + σ0, atoms) *
                M_loss *
                Hess_Φ(σ_dot_unit * t + σ0, atoms) *
                σ_dot_unit,
            -10,
            10,
        )
        return r[1]

    end

    for jj in eachindex(cs)
        cs[jj][ii] = Δ_pass(σ0, σ_dot, atom_sets[jj])
    end
end

## FIGURES
set_theme!(CF_theme)
colors = [CF_vermillion, CF_orange, CF_green, CF_sky]
fig = Figure(size = (1200, 1600))

supertitle = fig[1, 1]
Label(
    supertitle,
    "Traversing 3D unit cell",
    tellwidth = false,
    tellheight = false,
    font = :latex,
    fontsize = 36,
    valign = :center,
)

main_grid = fig[2:30, 1] = GridLayout()
legend_grid = fig[31, 1] = GridLayout()

square_data =
    load_object.([
        "Data/3D_Loss/3D_SoftShell_Square_Loss_λ1.0.jld2",
        "Data/3D_Loss/3D_SoftShell_Square_Loss_λ0.5.jld2",
        "Data/3D_Loss/3D_SoftShell_Square_Loss_λ0.25.jld2",
        "Data/3D_Loss/3D_SoftShell_Square_Loss_λ0.125.jld2",
    ])

cube_data =
    load_object.([
        "Data/3D_Loss/3D_SoftShell_Cube_Loss_λ1.0.jld2",
        "Data/3D_Loss/3D_SoftShell_Cube_Loss_λ0.5.jld2",
        "Data/3D_Loss/3D_SoftShell_Cube_Loss_λ0.25.jld2",
        "Data/3D_Loss/3D_SoftShell_Cube_Loss_λ0.125.jld2",
    ])

ax_square = Axis(
    main_grid[1, 1],
    # yscale = log10,
    title = "Square",
    ylabel = L"$\Delta/\dot{\sigma}_0\Phi_0^2$",
    xlabel = L"Initial speed $\dot{\sigma}_0$",
)
ax_cube = Axis(
    main_grid[2, 1],
    title = "Cube",
    # yscale = log10,
    ylabel = L"$\Delta/\dot{\sigma}_0\Phi_0^2$",
    xlabel = L"Initial speed $\dot{\sigma}_0$",
)
ax = [ax_square, ax_cube]
data = [square_data, cube_data]
labs = ["(a)", "(b)"]

for jj in eachindex(ax)

    text!(
        ax[jj],
        0.1,
        0.2,
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
            d[2],
            (d[3] ./ d[2]) ./ Φ0^2,
            color = colors[ii],
            marker = :cross,
            markersize = 20,
        )
        hlines!(ax[jj], [cs[jj][ii] ./ Φ0^2 |> real], color = colors[ii], lineswidth = 2)

    end
end

hidexdecorations!(ax_square)

polys = [PolyElement(color = c, strokecolor = :transparent) for c in colors]

Legend(
    legend_grid[1, 1],
    [polys],
    [["1", "1/2", "1/4", "1/8"]],
    [L"$\lambda$:"],
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
