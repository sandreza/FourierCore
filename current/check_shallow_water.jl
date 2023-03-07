using GLMakie, HDF5

filename = "shallow_water_large_ensemble"
directory = "/storage5/NonlocalPassiveTracers/Current/"
fid = h5open(directory * filename * ".hdf5", "r")

flux = Float64[]
flux_h = Float64[]
ktr = Float64[]
ph_speed = Float64[]
for case_number in 1:44
    push!(ktr, read(fid["kappa tracer $case_number"]))
    push!(ph_speed, read(fid["phase speed $case_number"]))
    push!(flux_h, read(fid["flux h $case_number"]))
    push!(flux, read(fid["flux tracer $case_number"]))
end

fig = Figure(resolution=(1100, 800))
ax = Axis(fig[1, 1]; xlabel="phase speed", ylabel="flux", xlabelsize=40, ylabelsize=40, xticklabelsize=30, yticklabelsize=30)
colors = [:red, :blue, :purple, :green]
options = (; linewidth=4)
labels = ["kappa = 0.01", "kappa = 0.05", "kappa = 0.15", "kappa = 0.2"]
for j in 1:4
    index_start = 1 + (j - 1) * 11
    index_end = 11 + (j - 1) * 11
    lines!(ax, ph_speed[index_start:index_end], flux[index_start:index_end]; color=colors[j], label=labels[j], options...)
    lines!(ax, ph_speed[1:11], flux_h[index_start:index_end]; color=(:grey, 0.5), options...)
end
axislegend(ax, position=:rt, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=100, labelsize=40)
ylims!(ax, (0.00, 0.12))
display(fig)

