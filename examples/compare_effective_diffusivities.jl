newhfile = h5open("effective_diffusivities.h5")

effective_diffusivity2 = read(newhfile["effective_diffusivities"]["3"])

close(newhfile)

fig = Figure()

ax = Axis(fig[1, 1]; title = "together", xlabel = "index", ylabel = "effective diffusivity")
ax2 = Axis(fig[1, 2]; title = "relative error percent", xlabel = "index", ylabel = "relative error")
scatter!(ax, effective_diffusivities)
scatter!(ax, effective_diffusivity2)
ylims!(ax, (0, 0.25))
scatter!(ax2, (effective_diffusivity2[1:29] ./ effective_diffusivities .- 1.0) * 100)