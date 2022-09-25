using HDF5, Statistics
using GLMakie 

fid = h5open("effective_diffusivities_samples.h5", "r")
N = length(fid["effective_diffusivities"])
M = length(fid["effective_diffusivities"]["1"])
amplitude_factor = zeros(N)
effective_diffusivities = zeros(M, N)
fourier_space_kernel = zeros(64, N)
for i in 1:N
    amplitude_factor[i] = read(fid["amplitude_factor"][string(i)])
    effective_diffusivities[:, i] .= read(fid["effective_diffusivities"][string(i)])
    fourier_space_kernel[2:32, i] .= effective_diffusivities[:, i]
    fourier_space_kernel[end-30:end, i] .= reverse(effective_diffusivities[:, i])
end

kernels = [circshift(real.(fft(fourier_space_kernel[:, i])), 32) for i in 1:N]
xs = (collect(0:64-1)  ./ 64) * 4π
kxs = 0.5:0.5:15.5

median_diffusivities = zeros(M)
maximum_diffusivities = zeros(M)
minimum_diffusivities = zeros(M)
for i in 1:M
    median_diffusivities[i] = median(effective_diffusivities[i, :])
    maximum_diffusivities[i] = maximum(effective_diffusivities[i, :])
    minimum_diffusivities[i] = minimum(effective_diffusivities[i, :])
end

##
fig = Figure()
ax = Axis(fig[1, 1], xlabel="wavenumber", ylabel="effective diffusivity", title = "samples")
ax2 = Axis(fig[1, 2], xlabel="wavenumber", ylabel="effective diffusivity", title = "max-min error bars and median dots")
ylims!(ax, (0, 0.25))
ylims!(ax2, (0, 0.25))
for i in 1:N
    scatter!(ax, kxs , effective_diffusivities[:, i], label = string(amplitude_factor[i]))
    scatter!(ax2, kxs , median_diffusivities, label = string(amplitude_factor[i]))
    errorbars!(ax2, kxs, median_diffusivities, median_diffusivities - minimum_diffusivities, maximum_diffusivities - median_diffusivities, whiskerwidth = 10)
end

close(fid)