using HDF5 
using GLMakie 

fid = h5open("effective_diffusivities.h5", "r")
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
##

fig = Figure()
ax = Axis(fig[1, 1], xlabel="wavenumber", ylabel="effective diffusivity")
ax2 = Axis(fig[1, 2]; xlabel = "|x-x'|", ylabel = "kernel")
xlims!(ax, (0, 16))
for i in 1:N
    scatter!(ax, kxs , effective_diffusivities[:, i], label = string(amplitude_factor[i]))
    kk = kernels[i][:]
    minkk = minimum(kk)
    lines!(ax2, xs, kk .- minkk, label = string(amplitude_factor[i]))
end
fig[1,3] = Legend(fig, ax2, "rms u₀", valign = :top)
display(fig)

close(fid)