using CairoMakie
using Colors
using ColorSchemes
using LaTeXStrings

## PLOTTING 
## Friendly colors
CF_red = colorant"rgba(204, 121, 167, 1.0)"
CF_vermillion = colorant"rgba(213, 94, 0, 1.0)"
CF_orange = colorant"rgba(230, 159, 0, 1.0)"
CF_yellow = colorant"rgba(240, 228, 66, 1.0)"
CF_green = colorant"rgba(0, 158, 115, 1.0)"
CF_sky = colorant"rgba(86, 180, 233, 1.0)"
CF_blue = colorant"rgba(0, 114, 178, 1.0)"
CF_black = colorant"rgba(0, 0, 0, 1.0)"
set_theme!()

CF_theme = Theme(
    fonts = (; latex = "CMU Serif", boldlatex = "CMU Serif Bold"),
    fontsize = 36,
    figure_padding = 10,
    Axis = (
        titlefont = :latex,
        xgridvisible = false,
        ygridvisible = false,
        xlabelfont = :latex,
        ylabelfont = :latex,
        xticklabelfont = :latex,
        yticklabelfont = :latex,
        xticklabelsize = 36,
        yticklabelsize = 36,
        titlesize = 36,
        xlabelsize = 36,
        ylabelsize = 36,
    ),
    Legend = (labelsize = 36, labelfont = :latex),
)

CF_vermillion_gradient = ColorScheme(range(colorant"white", CF_vermillion, length = 100))

CF_heat = ColorScheme(
    vcat(
        range(CF_black, CF_blue, length = 50),
        range(CF_blue, CF_vermillion, length = 150),
        range(CF_vermillion, CF_yellow, length = 150),
        range(CF_yellow, colorant"white", length = 50),
    ),
)
