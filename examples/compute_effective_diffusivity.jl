using GLMakie, JLD2, FFTW, Statistics
using FourierCore, FourierCore.Grid, FourierCore.Domain
# jld_name = "high_res_tracer_"
jld_name = "tracer_"
# WARNING!!!! due to bug need to multiply all θ̅a by 5/4
Ω = S¹(4π)^2
N = 2^8 # number of gridpoints
κ = 2 / 2^8 # fixed diffusivity # in general should load from file
grid = FourierGrid(N, Ω, arraytype = Array)
nodes, wavenumbers = grid.nodes, grid.wavenumbers
# in the future load from file
x = nodes[1]
y = nodes[2]
kˣ = wavenumbers[1]
kʸ = wavenumbers[2]

bugfix = 5/4
effective_diffusivity = Float64[]
index_choices = 1:81
for index_choice in index_choices
    println("on index ", index_choice)
    jlfile = jldopen(jld_name * string(index_choice) * ".jld2", "a+")
    ψ_save = jlfile["ψ_save"]
    θ_save = jlfile["θ_save"]
    θ̅a = mean(jlfile["θ̅a"], dims = 2)[:] * bugfix # messed up for tracer_ okay for others

    # calculate effective diffusivity
    # factor of 2 comes from the fact that the domain is 4π (as opposed to 2π)
    # will need to renormalize by length(θ̅a)/2 to take inverse transform
    if index_choice > 1
        𝒦ᵉᶠᶠ = ((length(θ̅a) / maximum(real.(fft(θ̅a)))) / 2) / (kˣ[index_choice]^2) - κ
    else
        𝒦ᵉᶠᶠ = abs(mean(θ̅a))
    end
    push!(effective_diffusivity, 𝒦ᵉᶠᶠ)
    close(jlfile)
end

fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "wavenumber", ylabel = "𝒦(k)", title = "Effective Diffusivity")
scatter!(ax, Array(kˣ[index_choices]), effective_diffusivity)

ylims!(ax, (0.0, 0.4))
xlims!(ax, (-0.1, kˣ[index_choices[end]+1]))
display(fig)
##
# fit model for effective diffusivity by picking out the pattern in the tail
tail_index = length(effective_diffusivity) - 1 # 19 was good
k₀ = kˣ[tail_index+1]
a = 1 / effective_diffusivity[tail_index]
c = (1 / effective_diffusivity[tail_index+1] - a) / (kˣ[tail_index+2]^2 - kˣ[tail_index+1]^2)
# model_𝒦ᵉᶠᶠ(k) = 1 / (a + c * (k^2 - k₀^2))
slope_index = index_choices[end] - 4
logk = log.(kˣ[2:length(effective_diffusivity)+1])
slope = (log.(effective_diffusivity[slope_index]) - log.(effective_diffusivity[slope_index-10])) / (logk[slope_index] - logk[slope_index-10])
fit_function(x) = slope * (x - logk[slope_index]) + log(effective_diffusivity[slope_index])

model_𝒦ᵉᶠᶠ(k) = abs(k) ≥ eps(100.0) ? exp(fit_function(log(abs(k)))) : 0.0 # just to handle zero well


lines!(ax, Array(kˣ[index_choices[2:end]]), model_𝒦ᵉᶠᶠ.(Array(kˣ[index_choices[2:end]])), color = :red)

##
hr_N = 1024 * 2
hr_grid = FourierGrid(hr_N, Ω, arraytype = Array)
hr_x = hr_grid.nodes[1]
hr_kˣ = hr_grid.wavenumbers[1]

fft_scaling = length(hr_x) / 2
wavenumberspace = model_𝒦ᵉᶠᶠ.(hr_kˣ[:]) * fft_scaling # copy(x[:]) .* 0.0
wavenumberspace[1] = effective_diffusivity[1] * fft_scaling
wavenumberspace[div(length(hr_x), 2)+1] = 0.0 # need this because of redefinition of kˣ

effective_diffusivity_mod = copy(effective_diffusivity[2:end])
NKeff = length(effective_diffusivity_mod)
wavenumberspace[2:1+NKeff] .= effective_diffusivity_mod * fft_scaling
wavenumberspace[length(hr_x)-NKeff+1:end] .= reverse(effective_diffusivity_mod) * fft_scaling
realspace = real.(ifft(wavenumberspace))

# realspace = real.(ifft([0.0, effective_diffusivity..., reverse(effective_diffusivity)...]))
index_shift = div(length(hr_x), 2) # div(length(realspace) - 1, 2) # div(length(x),2)
realspace_shifted = circshift(realspace, index_shift) # .- minimum(realspace)
println("zero'th mode is chosen to be ", minimum(realspace))
fig_kernel = Figure()
ax_kernel = Axis(fig_kernel[1, 1]; xlabel = "x-x'", ylabel = "𝒦(x-x')", title = "Effective Diffusivity")
lines!(ax_kernel, hr_x[:], realspace_shifted)


##
#=
fig = Figure()
ax = Axis(fig[1, 1])
logk = log.(kˣ[2:length(effective_diffusivity)+1])
scatter!(logk, log.(effective_diffusivity))
slope_index = index_choices[end] - 4
slope = (log.(effective_diffusivity[slope_index]) - log.(effective_diffusivity[slope_index-10]))/(logk[slope_index] - logk[slope_index-10] )
fit_function(x) = slope*(x - logk[slope_index]) + log(effective_diffusivity[slope_index])
lines!(logk, fit_function.(logk), color = :red, linewidth = 3)

fig = Figure()
ax = Axis(fig[1, 1])
logk = log.(kˣ[2:length(effective_diffusivity)+1])
scatter!(kˣ[2:length(effective_diffusivity)+1], effective_diffusivity)
fit_function(x) = -(x - 3) - 4
lines!(kˣ[2:length(effective_diffusivity)+1], exp.(fit_function.(logk)), color = :red, linewidth = 3)
=#



