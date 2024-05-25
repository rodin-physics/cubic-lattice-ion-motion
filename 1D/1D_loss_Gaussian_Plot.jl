include("../src/plotting.jl")
include("1D_loss_Gaussian.jl")

## FIGURES
set_theme!(CF_theme)
colors = [CF_vermillion, CF_orange, CF_green, CF_sky]
fig = Figure(size = (1200, 1200))

supertitle = fig[1, 1]
Label(
    supertitle,
    "Loss in 1D Lattice",
    tellwidth = false,
    tellheight = false,
    font = :latex,
    fontsize = 36,
    valign = :center,
)

main_grid = fig[2:20, 1] = GridLayout()
legend_grid = fig[21:22, 1] = GridLayout()

head_on_bound_data =
    load_object.([
        "Data/1D_Loss/1D_Gaussian_Loss_s0_λ0.5_ν1.jld2",
        "Data/1D_Loss/1D_Gaussian_Loss_s0_λ1.0_ν1.jld2",
        "Data/1D_Loss/1D_Gaussian_Loss_s0_λ2.0_ν1.jld2",
        "Data/1D_Loss/1D_Gaussian_Loss_s0_λ4.0_ν1.jld2",
    ])

offset_bound_data =
    load_object.([
        "Data/1D_Loss/1D_Gaussian_Loss_s1_λ0.5_ν1.jld2",
        "Data/1D_Loss/1D_Gaussian_Loss_s1_λ1.0_ν1.jld2",
        "Data/1D_Loss/1D_Gaussian_Loss_s1_λ2.0_ν1.jld2",
        "Data/1D_Loss/1D_Gaussian_Loss_s1_λ4.0_ν1.jld2",
    ])

head_on_unbound_data =
    load_object.([
        "Data/1D_Loss/1D_Gaussian_Loss_s0_λ0.5_ν0.jld2",
        "Data/1D_Loss/1D_Gaussian_Loss_s0_λ1.0_ν0.jld2",
        "Data/1D_Loss/1D_Gaussian_Loss_s0_λ2.0_ν0.jld2",
        "Data/1D_Loss/1D_Gaussian_Loss_s0_λ4.0_ν0.jld2",
    ])

offset_unbound_data =
    load_object.([
        "Data/1D_Loss/1D_Gaussian_Loss_s1_λ0.5_ν0.jld2",
        "Data/1D_Loss/1D_Gaussian_Loss_s1_λ1.0_ν0.jld2",
        "Data/1D_Loss/1D_Gaussian_Loss_s1_λ2.0_ν0.jld2",
        "Data/1D_Loss/1D_Gaussian_Loss_s1_λ4.0_ν0.jld2",
    ])

ax_head_on_bound = Axis(
    main_grid[1, 1],
    xlabel = "Head-on",
    xaxisposition = :top,
    ylabel = L"$\dot{\sigma}_0\Delta/\Phi_0^2$",
)
ax_offset_bound = Axis(
    main_grid[1, 2],
    xlabel = "Offset",
    xaxisposition = :top,
    ylabel = "Confined",
    yaxisposition = :right,
)
ax_head_on_unbound = Axis(
    main_grid[2, 1],
    ylabel = L"$\dot{\sigma}_0\Delta/\Phi_0^2$",
    xlabel = L"Initial speed $\dot{\sigma}_0$",
)
ax_offset_unbound = Axis(
    main_grid[2, 2],
    ylabel = "Unconfined",
    yaxisposition = :right,
    xlabel = L"Initial speed $\dot{\sigma}_0$",
)
ax = [ax_head_on_bound, ax_offset_bound, ax_head_on_unbound, ax_offset_unbound]
data = [head_on_bound_data, offset_bound_data, head_on_unbound_data, offset_unbound_data]
labs = ["(a)", "(b)", "(c)", "(d)"]

for jj in eachindex(ax)
    ylims!(ax[jj], -0.1, 1.75)
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
            d[4],
            d[4] .* d[5] ./ Φ0^2,
            color = colors[ii],
            marker = :cross,
            markersize = 12,
        )
        scatter!(
            ax[jj],
            d[4],
            d[4] .* d[6] ./ Φ0^2,
            color = colors[ii],
            marker = '□',
            markersize = 12,
        )
        lines!(ax[jj], d[7], d[7] .* d[8] ./ Φ0^2, color = colors[ii], linewidth = 2)

    end
end

hidexdecorations!(ax_head_on_bound, label = false)
hidexdecorations!(ax_offset_bound, label = false)

hideydecorations!(ax_offset_bound, label = false)
hideydecorations!(ax_offset_unbound, label = false)


polys = [PolyElement(color = c, strokecolor = :transparent) for c in colors]
sources = [
    MarkerElement(
        marker = '□',
        color = :black,
        strokecolor = :transparent,
        markersize = 16,
    ),
    MarkerElement(
        marker = :cross,
        color = :black,
        strokecolor = :transparent,
        markersize = 16,
    ),
    LineElement(color = :black, linewidth = 2),
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

save("1D_Loss.pdf", fig)
