using HDF5, GLMakie 

hfile = h5open("data_0.0_0.0_1.0_66.0_analysis.hdf5", "r")
samples = read(hfile["samples"])
data_cumulants = read(hfile["data cumulants"])
generative_cumulates = read(hfile["generative cumulants"])
close(hfile)

hfile = h5open("data_0.0_0.0_1.0_32.0.hdf5", "r")
snapshots = read(hfile["snapshots"] )
timeseries = read(hfile["timeseries"] )
snapshots_shape = read(hfile["snapshots shape"] )
timeseries_shape = read(hfile["timeseries shape"] )
close(hfile)
scale_factor = 2
r_snapshots = reshape(snapshots, (snapshots_shape[1], snapshots_shape[2], snapshots_shape[4], snapshots_shape[5])) * scale_factor;
##
fig = Figure(resolution = (1240, 556))
MM = 4
for i in 1:MM
    ax = Axis(fig[1,i]; title = "data")
    heatmap!(ax, r_snapshots[:, :, i, end], colorrange = (-2, 2), colormap = :balance, interpolate = true)
end
for i in 1:MM
    ax = Axis(fig[2,i]; title = "samples")
    heatmap!(ax, samples[:, :, 1, i] .* 2, colorrange = (-2, 2), colormap = :balance, interpolate = true)
end
ax = Axis(fig[1,1 + MM]; title = "data")
hist!(ax, r_snapshots[:], bins = 1000, normalization = :pdf)
xlims!(ax, (-3, 3))
ylims!(ax, (0, 0.7))
ax = Axis(fig[2,1 + MM]; title = "samples")
hist!(ax, samples[:] * 2, bins = 1000, normalization = :pdf)
xlims!(ax, (-3, 3))
ylims!(ax, (0, 0.7))
display(fig)