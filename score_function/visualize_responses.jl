using GLMakie, HDF5, LinearAlgebra

hfile = h5open("data_1.0_0.0_0.0_0.0_responses.hdf5", "r")
a_linear_response = read(hfile["linear response"])
a_hack_score_response = read(hfile["exact score response"])
a_numerical_response = read(hfile["perturbation response"])
close(hfile)
hfile = h5open("data_1.0_0.0_0.0_0.0_generative_response.hdf5", "r")
a_generative_response = read(hfile["generative response"])
close(hfile)

fig = Figure(resolution = (777, 233))
N = 32
lw = 3
ts = collect(0:99)
for i in 1:16
    indexchoice = i
    ii = (i-1)÷4 + 1
    jj = (i-1)%4 + 1
    ax = Axis(fig[ii, jj]; title = "1 -> " * string(indexchoice), xlabel = "time", ylabel = "response")
    lines!(ax, ts, a_linear_response[i, 1, :], color = (:blue, 0.4), linewidth = lw, label = "linear")
    lines!(ax, ts, a_hack_score_response[i, 1, :], color = (:red, 0.4), linewidth = lw, label = "score")
    lines!(ax, ts, a_generative_response[i, 1, 1:100], color = (:green, 0.4), linewidth = lw, label = "generative")
    scatter!(ax,ts, a_numerical_response[i, 1, :], color = (:orange, 0.4), linewidth = lw, label = "perturbation")
    if i == 1
        axislegend(ax, position = :rt)
    else
        hideydecorations!(ax)
    end
    xlims!(ax, (0, 100))
    ylims!(ax, (-0.15, 1.1))
end
display(fig)
[norm(a_numerical_response[i, 1, 1:50] - a_generative_response[i, 1, 1:50]) for i in 1:4]
[norm(a_numerical_response[i, 1, 1:50] - a_hack_score_response[i, 1, 1:50]) for i in 1:4]
[norm(a_numerical_response[i, 1, 1:50] - a_linear_response[i, 1, 1:50]) for i in 1:4]
save("nonlinear_responses.png", fig)

##
hfile = h5open("data_0.0_0.0_0.0_0.0_responses.hdf5", "r")
l_a_linear_response = read(hfile["linear response"])
l_a_hack_score_response = read(hfile["exact score response"])
l_a_numerical_response = read(hfile["perturbation response"])
close(hfile)
hfile = h5open("data_0.0_0.0_0.0_0.0_generative_response.hdf5", "r")
l_a_generative_response = read(hfile["generative response"])
close(hfile)

fig2 = Figure(resolution = (777, 233))
N = 32
lw = 3
ts = collect(0:99)
for i in 1:4
    indexchoice = i
    ii = (i-1)÷4 + 1
    jj = (i-1)%4 + 1
    ax = Axis(fig2[ii, jj]; title = "1 -> " * string(indexchoice), xlabel = "time", ylabel = "response")
    lines!(ax, ts, l_a_linear_response[i, 1, :], color = (:blue, 0.4), linewidth = lw, label = "linear")
    lines!(ax, ts, l_a_hack_score_response[i, 1, :], color = (:red, 0.4), linewidth = lw, label = "score")
    lines!(ax, ts, l_a_generative_response[i, 1, 1:100], color = (:green, 0.4), linewidth = lw, label = "generative")
    scatter!(ax,ts, l_a_numerical_response[i, 1, :], color = (:orange, 0.4), linewidth = lw, label = "perturbation")
    if i == 1
        axislegend(ax, position = :rt)
    else
        hideydecorations!(ax)
    end
    xlims!(ax, (0, 50))
    ylims!(ax, (-0.15, 1.1))
end
[norm(l_a_numerical_response[i, 1, 1:50] - l_a_generative_response[i, 1, 1:50]) for i in 1:4]
[norm(l_a_numerical_response[i, 1, 1:50] - l_a_hack_score_response[i, 1, 1:50]) for i in 1:4]
[norm(l_a_numerical_response[i, 1, 1:50] - l_a_linear_response[i, 1, 1:50]) for i in 1:4]
display(fig2)
save("linear_responses.png", fig2)

##

fig = Figure(resolution = (2*777, 2*233))
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

ts = collect(0:99)
for i in 1:4
    indexchoice = i
    ax = Axis(fig[2, i]; title = "1 -> " * string(indexchoice), xlabel = "time", ylabel = "response")
    lines!(ax, ts, l_a_linear_response[i, 1, :], color = (:blue, 0.4), linewidth = lw, label = "linear")
    lines!(ax, ts, l_a_hack_score_response[i, 1, :], color = (:red, 0.4), linewidth = lw, label = "score")
    lines!(ax, ts, l_a_generative_response[i, 1, 1:100], color = (:green, 0.4), linewidth = lw, label = "generative")
    scatter!(ax,ts, l_a_numerical_response[i, 1, :], color = (:orange, 0.4), linewidth = lw, label = "perturbation")
    if i == 1
        axislegend(ax, position = :rt)
    else
        hideydecorations!(ax)
    end
    xlims!(ax, (0, 50))
    ylims!(ax, (-0.15, 1.1))
end
display(fig)
##
save("together_responses.png", fig)