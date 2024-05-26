include("3D_loss_Gaussian_lattice.jl")
include("../src/plotting.jl")

atoms = [(1, 1, 1), (1, 2, 1), (2, 1, 1), (2, 2, 1)]
c_square = [Δ_pass([α / 2, α / 2, 0], [0, 0, 1], λ, atoms, α) for λ in λs]

atoms =
    [(1, 1, 1), (1, 2, 1), (2, 1, 1), (2, 2, 1), (1, 1, 2), (1, 2, 2), (2, 1, 2), (2, 2, 2)]
c_cube = [Δ_pass([α / 2, α / 2, 0], [0, 0, 1], λ, atoms, α) for λ in λs]

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
        "Data/3D_Loss/3D_Gaussian_Square_Loss_λ0.5.jld2",
        "Data/3D_Loss/3D_Gaussian_Square_Loss_λ1.0.jld2",
        "Data/3D_Loss/3D_Gaussian_Square_Loss_λ2.0.jld2",
        "Data/3D_Loss/3D_Gaussian_Square_Loss_λ4.0.jld2",
    ])

cube_data =
    load_object.([
        "Data/3D_Loss/3D_Gaussian_Cube_Loss_λ0.5.jld2",
        "Data/3D_Loss/3D_Gaussian_Cube_Loss_λ1.0.jld2",
        "Data/3D_Loss/3D_Gaussian_Cube_Loss_λ2.0.jld2",
        "Data/3D_Loss/3D_Gaussian_Cube_Loss_λ4.0.jld2",
    ])

ax_square = Axis(
    main_grid[1, 1],
    yscale = log10,
    title = "Square",
    ylabel = L"$\Delta/\dot{\sigma}_0\Phi_0^2$",
    xlabel = L"Initial speed $\dot{\sigma}_0$",
)
ax_cube = Axis(
    main_grid[2, 1],
    title = "Cube",
    yscale = log10,
    ylabel = L"$\Delta/\dot{\sigma}_0\Phi_0^2$",
    xlabel = L"Initial speed $\dot{\sigma}_0$",
)
ax = [ax_square, ax_cube]
data = [square_data, cube_data]
cs = [c_square, c_cube]
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

        lines!(ax[jj], d[2], cs[jj][ii] * ones(nPts) |> real, color = colors[ii], lineswidth = 2)
    end
end

hidexdecorations!(ax_square)

polys = [PolyElement(color = c, strokecolor = :transparent) for c in colors]

Legend(
    legend_grid[1, 1],
    [polys],
    [["1/2", "1", "2", "4"]],
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

save("3D_Loss_Lattice.pdf", fig)
