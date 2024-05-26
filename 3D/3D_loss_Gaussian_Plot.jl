include("../src/plotting.jl")
include("3D_loss_Gaussian.jl")

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
        "Data/3D_Loss/3D_Gaussian_Loss_s0_λ0.5.jld2",
        "Data/3D_Loss/3D_Gaussian_Loss_s0_λ1.0.jld2",
        "Data/3D_Loss/3D_Gaussian_Loss_s0_λ2.0.jld2",
        "Data/3D_Loss/3D_Gaussian_Loss_s0_λ4.0.jld2",
    ])

offset_data =
    load_object.([
        "Data/3D_Loss/3D_Gaussian_Loss_s1_λ0.5.jld2",
        "Data/3D_Loss/3D_Gaussian_Loss_s1_λ1.0.jld2",
        "Data/3D_Loss/3D_Gaussian_Loss_s1_λ2.0.jld2",
        "Data/3D_Loss/3D_Gaussian_Loss_s1_λ4.0.jld2",
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
        c = Δ_pass([0, s, 0], [0, 0, 1], λ, atoms, α) |> real
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
