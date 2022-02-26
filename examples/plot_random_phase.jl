using GLMakie

time_index = Observable(1)
xa = Array(x[:])
ya = Array(y[:])
θ̅a = Array(real.(θ̅))
ψfield = @lift(ψ_save[$time_index])
θfield = @lift(θ_save[$time_index])
fig = Figure(resolution = (1722, 1076))
ax = Axis(fig[1, 1]; title = "stream function ")
ax2 = Axis(fig[1, 2]; title = "tracer concentration")
ax3 = Axis(fig[1, 3]; title = "tracer concentration time average")
heatmap!(ax, xa, ya, ψfield, interpolate = true, colormap = :balance, colorrange = (-1.5, 1.5))
heatmap!(ax2, xa, ya, θfield, interpolate = true, colormap = :balance, colorrange = θclims ./3)
heatmap!(ax3, xa, ya, θ̅a, interpolate = true, colormap = :balance,)

display(fig)
time_index[] = length(ψ_save)



