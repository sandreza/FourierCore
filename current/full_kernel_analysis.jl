using HDF5, GLMakie, FFTW 

directory = "/storage5/NonlocalPassiveTracers/Current/"
ii = 3 # forcing, 3, 4
kk = 1 # hypo, 1, 2
jj = 1 # hyper
base_name = "even_higher_frequency_general_case_"
filename = base_name * string(ii) * "_" * string(jj) * "_" * string(kk) * ".hdf5"

hfile = h5open(directory * filename, "r")
fgf = read(hfile["fourier space kernel"])
ωs = reverse(read(hfile["time dependent frequences"]))
kernel = read(hfile["space time kernel"])
close(hfile)


##
kernel3 = copy(kernel)
# kernel3 = copy(-circshift(real.(ifft(fgf)), (41, 0)))
Tfundamental = 2π / ωs[2]
plotting_indices = collect(1:50)
times = Tfundamental / (size(kernel3)[2]-1) * (plotting_indices .- 1)
nK = size(kernel3)[1]
xs = collect(0:nK-1) / nK .* 2π
fig = Figure(resolution = (1400, 800))
ax11 = Axis(fig[1,1]; title = "kernel for different time lags")
op = 0.5
colorlist = [(:red, op), (:blue, op), (:green, op), (:orange, op), (:purple, op)]
lw = 3
for (j, i) in enumerate([1, 8+1, 16+1, 24+1, 32+1])
    lines!(ax11, xs, kernel3[:, plotting_indices[i]], label = string("t = ") * string(times[i]), color = colorlist[j], linewidth = lw)
end
axislegend(ax11, position=:rt, framecolor=(:grey, 0.5), patchsize=(30, 30), markersize=50, labelsize=20)

ax12 = Axis(fig[1,2]; title = "∫dt' kernel", xlabel = "x")
scatter!(ax12, xs, sum(kernel3, dims =2)[:] .* times[2])

ax21 = Axis(fig[2,1]; title = "space time kernel", ylabel = "time", xlabel = "space")
ts = Tfundamental / (size(kernel3)[2]-1) * collect(0:20)
heatmap!(ax21, xs, ts, kernel3[:, 1:21])

ts = Tfundamental / (size(kernel3)[2]-1) * collect(0:100)
ax22 = Axis(fig[2,2]; title = "kernel peak as a function of time", ylabel = "peak", xlabel = "time")
scatter!(ax22, ts, [maximum(kernel3[:, i]) for i in eachindex(ts)], label = "peak")
display(fig)