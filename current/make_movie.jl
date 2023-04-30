
vorticities = typeof(real.(Array(ζ))[:, :, 1])[]
iter = ProgressBar(1:15000)
global tval = 0.0
ts = typeof(tval)[]
mod_index = 10
for i = iter
    step!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, t, parameters)
    global tval += Δt
    if i % mod_index == 0
        push!(vorticities, real.(Array(ζ))[:, :, 1])
        push!(ts, tval)
    end
end
##
using GLMakie

fig = Figure(resolution=(2400, 600))
sl_x = Slider(fig[2, 1], range=1:length(vorticities), startvalue=1)
o_index = sl_x.value
field = @lift vorticities[$o_index]
titlestring = @lift( "t = " * string(ts[$o_index]))
ax = Axis(fig[1, 1]; title = titlestring)
heatmap!(ax, field, colormap=:viridis, colorrange = (-1,1))
display(fig)

#=
framerate = 30
timestamps = 1:length(vorticities)
GLMakie.record(fig, "time_animation_vorticity.mp4", timestamps;
    framerate=framerate) do t
    o_index[] = t
    println("done with frame $t")
    nothing
end;
=#