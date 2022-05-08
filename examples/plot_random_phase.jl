using GLMakie, JLD2, FFTW
jld_name = "delete_me_later"
# jld_name = "high_order_timestep_spatial_tracer_"
index_choice = 2
fig = Figure(resolution = (1722, 1076))
effective_diffusivity = Float64[]
#=
for index_choice in 1:1

    jlfile = jldopen("tracer_" * string(index_choice) * ".jld2", "a+")
    ψ_save = jlfile["ψ_save"]
    θ_save = jlfile["θ_save"]
    θ̅a = jlfile["θ̅a"]

    θ_l_infty = maximum(abs.(θ_save[end][:]))
    θ_saveclims = (-θ_l_infty, θ_l_infty)

    θ̅_l_infty = maximum(abs.(θ̅a))
    θ̅_saveclims = (-θ̅_l_infty, θ̅_l_infty)

    time_index = Observable(1)
    xa = Array(x[:])
    ya = Array(y[:])
    ψfield = @lift(ψ_save[$time_index])
    θfield = @lift(θ_save[$time_index])
    ax = Axis(fig[index_choice-1, 1]; title = "stream function ")
    ax2 = Axis(fig[index_choice-1, 2]; title = "tracer concentration")
    ax3 = Axis(fig[index_choice-1, 3]; title = "tracer concentration time average")
    heatmap!(ax, xa, ya, ψfield, interpolate = true, colormap = :balance, colorrange = (-1.5, 1.5))
    heatmap!(ax2, xa, ya, θfield, interpolate = true, colormap = :balance, colorrange = θ_saveclims)
    heatmap!(ax3, xa, ya, θ̅a, interpolate = true, colormap = :balance, colorrange = θ̅_saveclims)
    close(jlfile)
    #=
    jlfile = jldopen("tracer_" * string(index_choice) * ".jld2", "a+")
    ψ_save = jlfile["ψ_save"]
    θ_save = jlfile["θ_save"]
    θ̅a = jlfile["θ̅a"]

    θ_l_infty = maximum(abs.(θ_save[end][:]))
    θ_saveclims = (-θ_l_infty, θ_l_infty)

    θ̅_l_infty = maximum(abs.(θ̅a))
    θ̅_saveclims = (-θ̅_l_infty, θ̅_l_infty)
    fig = Figure(resolution = (1722, 1076))
    ax = Axis(fig[1, 1]; title = "stream function ")
    ax2 = Axis(fig[1, 2]; title = "tracer concentration")
    ax3 = Axis(fig[1, 3]; title = "tracer concentration time average")
    time_index = Observable(10)
    ψfield = @lift(ψ_save[$time_index])
    θfield = @lift(θ_save[$time_index])
    heatmap!(ax, ψfield, interpolate = true, colormap = :balance, colorrange = (-1.5, 1.5))
    heatmap!(ax2, θfield, interpolate = true, colormap = :balance, colorrange = θ_saveclims)
    heatmap!(ax3, θ̅a, interpolate = true, colormap = :balance, colorrange = θ̅_saveclims)
    =#

    # calculate effective diffusivity
    # will need to renormalize by length(θ̅a)/2 to take inverse transform
    𝒦ᵉᶠᶠ = ((length(θ̅a) / maximum(real.(fft(θ̅a)))) / 2) / (kˣ[index_choice]^2) - κ
    push!(effective_diffusivity, 𝒦ᵉᶠᶠ)
end
=#

jlfile = jldopen(jld_name * string(index_choice) * ".jld2", "a+")
ψ_save = jlfile["ψ_save"]
θ_save = jlfile["θ_save"]
xa = jlfile["xnodes"]
ya = jlfile["ynodes"]
θ̅a = jlfile["θ̅a"]
kˣ = jlfile["kˣ_wavenumbers"]
keff = ((length(θ̅a) / maximum(real.(fft(θ̅a)))) / 2) / (kˣ[index_choice]^2)
println("the effective diffusivty is ", keff)

θ_l_infty = maximum(abs.(θ_save[end][:]))
θ_saveclims = (-θ_l_infty, θ_l_infty)

θ̅_l_infty = maximum(abs.(θ̅a))
θ̅_saveclims = (-θ̅_l_infty, θ̅_l_infty)

time_index = Observable(1)
ψfield = @lift(ψ_save[$time_index])
θfield = @lift(θ_save[$time_index])
ax = Axis(fig[1, 1]; title = "stream function ")
ax2 = Axis(fig[1, 2]; title = "tracer concentration")
ax3 = Axis(fig[1, 3]; title = "tracer concentration time average")
heatmap!(ax, xa, ya, ψfield, interpolate = true, colormap = :balance, colorrange = (-1.5, 1.5))
heatmap!(ax2, xa, ya, θfield, interpolate = true, colormap = :balance, colorrange = θ_saveclims)
heatmap!(ax3, xa, ya, θ̅a, interpolate = true, colormap = :balance, colorrange = θ̅_saveclims)
close(jlfile)

display(fig)
time_index[] = 10






