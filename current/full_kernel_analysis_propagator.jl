using GLMakie, HDF5

directory = "/storage5/NonlocalPassiveTracers/Current/"
base_name = "full_propagator_"
ii = 3 # forcing
kk = 1 # hypo
jj = 1 # hyper
filename = base_name * string(ii) * "_" * string(jj) * "_" * string(kk)

fid = h5open(directory * filename * ".hdf5", "r")
𝒦 = read(fid["space time kernel"])
ts = read(fid["space time kernel timelist"])
close(fid)

##
tindlist = [1, 64+1, 128+1, 192+1, 256+1]# [128+1, 256+1, 512+1, 768+1, 1024+1]
timelist = [0, 0.25, 0.5, 0.75, 1.0]# [0.5, 1, 2, 3, 4]
op = 0.5
colorlist = [(:red, op), (:blue, op), (:green, op), (:orange, op), (:purple, op)]
xs = (collect(0:127)/128 * 4π) .- 2π
fig = Figure(resolution = (1400, 800))
ax11 = Axis(fig[1,1]; title = "kernel for different time lags")
lw = 3
for (j, i) in enumerate(tindlist)
    lines!(ax11, xs, 𝒦[:, i], label = string("t = ") * string(timelist[j]), color = colorlist[j], linewidth = lw)
end
axislegend(ax11, position=:rt, framecolor=(:grey, 0.5), patchsize=(30, 30), markersize=50, labelsize=20)

ax12 = Axis(fig[1,2]; title = "∫dt' kernel", xlabel = "x")
scatter!(ax12, xs, sum(𝒦, dims =2)[:] * Δt)

ax21 = Axis(fig[2,1]; title = "space time kernel", ylabel = "time", xlabel = "space")
heatmap!(ax21, xs, ts[128+1:1024+1], 𝒦[:, 128+1:1024+1])

ax22 = Axis(fig[2,2]; title = "kernel peak as a function of time", ylabel = "peak", xlabel = "time")
scatter!(ax22, ts[1:1024+1], [maximum(𝒦[:, i]) for i in 1:1024+1], label = "peak")
display(fig)

##
r𝒦 = (𝒦[1:2:end, :] + 𝒦[2:2:end, :])/2

tindlist = [1, 64+1, 128+1, 192+1, 256+1]# [128+1, 256+1, 512+1, 768+1, 1024+1]
timelist = [0, 0.25, 0.5, 0.75, 1.0]# [0.5, 1, 2, 3, 4]
op = 0.5
colorlist = [(:red, op), (:blue, op), (:green, op), (:orange, op), (:purple, op)]
rxs = (collect(0:63)/64 * 4π) .- 2π
fig = Figure(resolution = (1400, 800))
ax11 = Axis(fig[1,1]; title = "kernel for different time lags")
lw = 3
for (j, i) in enumerate(tindlist)
    lines!(ax11, rxs, r𝒦[:, i], label = string("t = ") * string(timelist[j]), color = colorlist[j], linewidth = lw)
end
axislegend(ax11, position=:rt, framecolor=(:grey, 0.5), patchsize=(30, 30), markersize=50, labelsize=20)

ax12 = Axis(fig[1,2]; title = "∫dt' kernel", xlabel = "x")
scatter!(ax12, rxs, sum(r𝒦, dims =2)[:] * Δt)

ax21 = Axis(fig[2,1]; title = "space time kernel", ylabel = "time", xlabel = "space")
heatmap!(ax21, rxs, ts[128+1:1024+1], r𝒦[:, 128+1:1024+1])

ax22 = Axis(fig[2,2]; title = "kernel peak as a function of time", ylabel = "peak", xlabel = "time")
scatter!(ax22, ts[1:1024+1], [maximum(r𝒦[:, i]) for i in 1:1024+1], label = "peak")
display(fig)