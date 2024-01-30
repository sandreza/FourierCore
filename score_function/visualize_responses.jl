using GLMakie, HDF5

hfile = h5open("data_0.0_0.0_1.0_66.0_responses.hdf5", "r")
a_linear_response = read(hfile["linear response"])
a_hack_score_response = read(hfile["exact score response"])
a_numerical_response = read(hfile["perturbation response"])
a_generative_response = read(hfile["generative response"])
close(hfile)

fig = Figure(resolution = (777, 233))
N = 32
lw = 3
ts = collect(0:99)
for i in 1:4
    indexchoice = i
    ii = (i-1)÷4 + 1
    jj = (i-1)%4 + 1
    ax = Axis(fig[ii, jj]; title = "1 -> " * string(indexchoice), xlabel = "time", ylabel = "response")
    lines!(ax, ts, a_linear_response[i, 1, :], color = (:blue, 0.4), linewidth = lw, label = "linear")
    lines!(ax, ts, a_hack_score_response[i, 1, :], color = (:red, 0.4), linewidth = lw, label = "score")
    lines!(ax, ts, a_generative_response[i, 1, :], color = (:green, 0.4), linewidth = lw, label = "generative")
    scatter!(ax,ts, a_numerical_response[i, 1, :], color = (:orange, 0.4), linewidth = lw, label = "perturbation")
    if i == 1
        axislegend(ax, position = :rt)
    else
        hideydecorations!(ax)
    end
    xlims!(ax, (0, 50))
    ylims!(ax, (-0.15, 1.1))
end
display(fig)
save("responses.png", fig)