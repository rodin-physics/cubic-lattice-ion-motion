include("../3D/3D_Gaussian_time_local_dissipation.jl")

(pos, speed, pos_particle, speed_particle, δτ, Φ0, λ) =
    load_object("Data/3D_Loss/Local_Loss.jld2")
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
