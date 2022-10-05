using HDF5, FFTW

hfile = h5open("effective_diffusivities.h5", "r")

index = 3
amp_factor = read(hfile["amplitude_factor"][string(index)])
effective_diffusivity = read(hfile["effective_diffusivities"][string(index)])
wavenumber_index = collect(1:length(effective_diffusivity))

## Assume function form effective_diffusivity = a * (wavenumber_index)^d
## log(effective_diffusivity) = log(a) + d * log(wavenumber_index)

log_effective_diffusivity = log.(effective_diffusivity)
log_wavenumber_index = log.(wavenumber_index)
Δlog_effective_diffusivity = log_effective_diffusivity[2:end] - log_effective_diffusivity[1:end-1]
Δlog_wavenumber_index = log.(wavenumber_index[2:end]) - log.(wavenumber_index[1:end-1])
log_slope = Δlog_effective_diffusivity ./ Δlog_wavenumber_index
log_axis = (log_effective_diffusivity[2:end] + log_effective_diffusivity[1:end-1]) / 2 .- log_slope .* (log.(wavenumber_index)[1:end-1] + log.(wavenumber_index)[2:end]) / 2
endex = maximum([30, length(Δlog_effective_diffusivity)])
effective_diffusivity_index_fit(i; endex=length(log_effective_diffusivity) - 1) = exp(log_axis[endex] + log_slope[endex] * log(i))
# endex = extrapolation index

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, wavenumber_index, effective_diffusivity, label="effective diffusivity")
lines!(ax, wavenumber_index, effective_diffusivity_index_fit.(wavenumber_index), label="fit")

##
# Now use the extrapolation
extrapolation_grid_length = 128
symindex = floor(Int, extrapolation_grid_length / 2) - 1
extrapolated_effective_diffusivity = effective_diffusivity_index_fit.(1:symindex)
extrapolated_effective_diffusivity[1:length(effective_diffusivity)] .= effective_diffusivity

effective_diffusivity_list = zeros(extrapolation_grid_length)
effective_diffusivity_list[2:symindex+1] .= extrapolated_effective_diffusivity
effective_diffusivity_list[end-symindex+1:end] .= reverse(extrapolated_effective_diffusivity)
kernel = real.(fft(effective_diffusivity_list))
effective_diffusivity_list = real.(ifft(kernel .- minimum(kernel)))

##
# Now create operator under the assumption that the effective diffusivity is isotropic
effective_diffusivity_operator = zeros(extrapolation_grid_length, extrapolation_grid_length)
half_grid_length = floor(Int, extrapolation_grid_length / 2)
for i in 1:half_grid_length, j in 1:half_grid_length
    if i^2 + j^2 >= half_grid_length^2
        # note that the fit operator is off by one
        effective_diffusivity_operator[i, j] = effective_diffusivity_index_fit(sqrt((i - 1)^2 + (j - 1)^2) + 1 + 1)
    else
        index_magnitude = sqrt((i - 1)^2 + (j - 1)^2) + 1
        upindex = ceil(Int, index_magnitude)
        downindex = floor(Int, index_magnitude)
        weight = upindex - index_magnitude
        effective_diffusivity_operator[i, j] = effective_diffusivity_list[upindex] * (1 - weight) + effective_diffusivity_list[downindex] * weight
    end

end
effective_diffusivity_operator[end-symindex+1:end, :] .= reverse(effective_diffusivity_operator[2:symindex+1, :], dims=1)
effective_diffusivity_operator[:, end-symindex+1:end] .= reverse(effective_diffusivity_operator[:, 2:symindex+1], dims=2)
