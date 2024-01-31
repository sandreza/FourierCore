using HDF5, GLMakie 

hfile = h5open("data_1.0_0.0_0.0_0.0_analysis.hdf5", "r")
samples = read(hfile["samples"])
data_cumulants = read(hfile["data cumulants"])
generative_cumulates = read(hfile["generative cumulants"])
close(hfile)

hfile = h5open("data_1.0_0.0_0.0_0.0.hdf5", "r")
snapshots = read(hfile["snapshots"] )
timeseries = read(hfile["timeseries"] )
snapshots_shape = read(hfile["snapshots shape"] )
timeseries_shape = read(hfile["timeseries shape"] )
scale_factor = read(hfile["scale factor"] )
close(hfile)

r_snapshots = reshape(snapshots, (snapshots_shape[1], snapshots_shape[2], snapshots_shape[4], snapshots_shape[5])) * scale_factor;
r_timeseries = reshape(timeseries, (timeseries_shape[1], timeseries_shape[2], timeseries_shape[4], timeseries_shape[5])) * scale_factor;
##
fig = Figure(resolution = (1240, 556))
MM = 4
for i in 1:MM
    ax = Axis(fig[1,i]; title = "data")
    heatmap!(ax, r_snapshots[:, :, i, end] .* scale_factor , colorrange = (-2, 2), colormap = :balance, interpolate = true)
end
for i in 1:MM
    ax = Axis(fig[2,i]; title = "samples")
    heatmap!(ax, samples[:, :, 1, i] * scale_factor^2, colorrange = (-2, 2), colormap = :balance, interpolate = true)
end
ax = Axis(fig[1,1 + MM]; title = "data")
hist!(ax, r_snapshots[:]  .* scale_factor, bins = 1000, normalization = :pdf)
xlims!(ax, (-3, 3))
ylims!(ax, (0, 0.7))
ax = Axis(fig[2,1 + MM]; title = "samples")
hist!(ax, samples[:]  * scale_factor^2 ,  bins = 1000, normalization = :pdf)
xlims!(ax, (-3, 3))
ylims!(ax, (0, 0.7))
display(fig)
save("samples_vs_data.png", fig)
##
fig = Figure(resolution = (800, 450))
for i in 1:8
    indexchoice = i
    ii = (i-1)÷4 + 1
    jj = (i-1)%4 + 1
    skip_index = 10 * (i-1) + 1 
    ax = Axis(fig[ii, jj]; title = "t = $(skip_index-1)")
    heatmap!(ax, r_timeseries[:, :, 31, skip_index] * scale_factor, colorrange = (-2, 2), colormap = :balance, interpolate = true)
    hidedecorations!(ax)
end
display(fig)
save("timeseries.png", fig)