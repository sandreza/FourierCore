fig = Figure(resolution=(1400, 1100))
ax11 = Axis(fig[1, 1]; title="ensemble average")
ax12 = Axis(fig[1, 2]; title="x=0 slice")
ax21 = Axis(fig[2, 1]; title="diffusion")
ax22 = Axis(fig[2, 2]; title="nonlocal space kernel")

t_slider = Slider(fig[3, 1:2], range=1:iend, startvalue=0)
tindex = t_slider.value
colormap = :bone_1
field = @lift(Array(θ̅_timeseries[:, :, $tindex]))
field_slice = @lift($field[:, floor(Int, N / 2)])
# field_slice = @lift(mean($field[:, :], dims=2)[:])
# colorrange=(0.0, 1.0), 
Δ_A = Array(Δ)

colorrange = @lift((0, maximum($field)))
field_diffusion = @lift(real.(ifft(fft(θ_A) .* exp.(Δ_A * ($tindex - 0) * effective_diffusivity[2] * Δt))))
field_diffusion_slice = @lift($field_diffusion[:, floor(Int, N / 2)])
# field_diffusion_slice = @lift(mean($field_diffusion[:, :], dims = 2)[:])


KK = (effective_diffusivity_operator .+ κ) .* Δ_A
approximate_field = @lift(real.(ifft(fft(θ_A) .* exp.(KK * ($tindex - 0) * Δt))))
approximate_field_slice = @lift($approximate_field[:, floor(Int, N / 2)])
# approximate_field_slice = @lift(mean($approximate_field[:, :], dims = 2)[:])

heatmap!(ax11, x_A, x_A, field, colormap=colormap, interpolate=true, colorrange=colorrange)
heatmap!(ax21, x_A, x_A, field_diffusion, colormap=colormap, interpolate=true, colorrange=colorrange)
heatmap!(ax22, x_A, x_A, approximate_field, colormap=colormap, interpolate=true, colorrange=colorrange)

le = lines!(ax12, x_A, field_slice, color=:black)
ld = lines!(ax12, x_A, field_diffusion_slice, color=:red)
lnd = lines!(ax12, x_A, approximate_field_slice, color=:blue)

axislegend(ax12, [le, ld, lnd], ["ensemble", "diffusion", "nonlocal space"], position=:rt)

display(fig)

##
#=
tmp = [real.(fft(Array(θ̅_timeseries[:, floor(Int, N / 2), i]))) for i in 1:iend]

begin
fig = Figure()
ax1 = Axis(fig[1,1])
for j in 19:25
    ttmp = [tmp[i][j] for i in 1:iend]
    lines!(ax1, ttmp ./ ttmp[1] )
end
display(fig)
end
=#
