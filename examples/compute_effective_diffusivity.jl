using GLMakie, JLD2, FFTW
using FourierCore, FourierCore.Grid, FourierCore.Domain

Ω = S¹(4π)^2
N = 2^8 # number of gridpoints
κ = 2 / N
grid = FourierGrid(N, Ω, arraytype = Array)
nodes, wavenumbers = grid.nodes, grid.wavenumbers

x = nodes[1]
y = nodes[2]
kˣ = wavenumbers[1]
kʸ = wavenumbers[2]


effective_diffusivity = Float64[]
index_choices = 2:20
for index_choice in index_choices
    println("on index ", index_choice)
    jlfile = jldopen("tracer_" * string(index_choice) * ".jld2", "a+")
    ψ_save = jlfile["ψ_save"]
    θ_save = jlfile["θ_save"]
    θ̅a = jlfile["θ̅a"]

    # calculate effective diffusivity
    # factor of 2 comes from the fact that the domain is 4π (as opposed to 2π)
    # will need to renormalize by length(θ̅a)/2 to take inverse transform
    𝒦ᵉᶠᶠ = ((length(θ̅a) / maximum(real.(fft(θ̅a)))) / 2) / (kˣ[index_choice]^2) - κ
    push!(effective_diffusivity, 𝒦ᵉᶠᶠ)
    close(jlfile)
end

fig = Figure()
ax = Axis(fig[1, 1]; xlabel = "wavenumber", ylabel = "𝒦(k)", title = "Effective Diffusivity")
scatter!(ax, Array(kˣ[index_choices]), effective_diffusivity)

ylims!(ax, (0.0, 0.4))
xlims!(ax, (0.0, kˣ[index_choices[end]+1]))
display(fig)
##
# fit model of the form 1/𝒦ᵉᶠᶠ(k) = a + c * (k^2 - k₀^2)
tail_index = 18
k₀ = kˣ[tail_index+1]
a = 1 / effective_diffusivity[tail_index]
c = (1 / effective_diffusivity[tail_index+1] - a) / (kˣ[tail_index+2]^2 - kˣ[tail_index+1]^2)
model_𝒦ᵉᶠᶠ(k) = 1 / (a + c * (k^2 - k₀^2))

lines!(ax, Array(kˣ[index_choices]), model_𝒦ᵉᶠᶠ.(Array(kˣ[index_choices])), color = :red)

wavenumberspace = model_𝒦ᵉᶠᶠ.(kˣ[:]) * length(x) / 2 # copy(x[:]) .* 0.0
wavenumberspace[1] = 0.0
wavenumberspace[div(length(x),2)+1] = 0.0 # need this because of redefinition of kˣ

effective_diffusivity_mod = copy(effective_diffusivity)
NKeff = length(effective_diffusivity_mod)
wavenumberspace[2:1+NKeff] .= effective_diffusivity_mod * length(x) / 2
wavenumberspace[length(x)-NKeff+1:end] .= reverse(effective_diffusivity_mod) * length(x) / 2
realspace = real.(ifft(wavenumberspace))

# realspace = real.(ifft([0.0, effective_diffusivity..., reverse(effective_diffusivity)...]))
index_shift = div(length(x), 2) # div(length(realspace) - 1, 2) # div(length(x),2)
realspace_shifted = circshift(realspace, index_shift) .- minimum(realspace)
scatter(realspace_shifted)
