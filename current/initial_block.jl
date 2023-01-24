using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2
# using GLMakie, HDF5
using ProgressBars
rng = MersenneTwister(1234)
Random.seed!(123456789)

# initialize fields: variables and domain defined here
include("initialize_fields.jl")

amplitude_factor = 1.0 # normalized later
phase_speed = 1.0
# number of gridpoints in transition is about λ * N / 2
bump(x; λ=20 / N[1], width=π/2 ) = 0.5 * (tanh((x + width / 2) / λ) - tanh((x - width / 2) / λ))

r_A = Array(@. sqrt((x - 2π)^2 + (y - 2π)^2))
θ_A = [bump(r_A[i, j]) for i in 1:N[1], j in 1:N[1]]
θ .= CuArray(θ_A)
P * θ # in place fft
@. 𝒟θ = 𝒟κ * θ
P⁻¹ * 𝒟θ # in place fft
sθ .= -𝒟θ
P⁻¹ * θ # in place fft

t = [0.0]
tend = 1
iend = ceil(Int, tend / Δt)

# new realization of flow
rand!(rng, φ) # between 0, 1
φ .*= 2π # to make it a random phase

load_psi!(ψ)
ζ .= ifft(Δ .* fft(ψ))

θ .= CuArray(θ_A)
iter = ProgressBar(1:iend)
for i = iter
    step!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, parameters)
    if i % 10 == 0
        θ_min, θ_max = extrema(real.(θ))
        ζ_min, ζ_max = extrema(real.(ζ))
        set_multiline_postfix(iter, "θ_min: $θ_min \nθ_max: $θ_max \nζ_min: $ζ_min \nζ_max: $ζ_max")
    end
end



x_A = Array(x)[:] .- 2π
θ_F = Array(real.(θ[:,:,24]))
θ̅_F = Array(mean(real.(θ), dims=3))[:,:]

fig = Figure()
for i in 1:25
    ii = (i-1) ÷ 5 + 1
    jj = (i-1) % 5 + 1
    ax = Axis(fig[ii, jj], title="realization $i")
    heatmap!(ax, x_A, x_A, real.(Array(θ)[:,:,i]), colormap=:bone_1, colorrange=(0.0, 1.0), interpolate=true)
end

begin
    fig = Figure(resolution=(2048, 512))
    ax1 = Axis(fig[1, 1], title="t = 0")
    ax2 = Axis(fig[1, 2], title="instantaneous t = " * string(tend))
    ax3 = Axis(fig[1, 4], title="ensemble average t = " * string(tend))
    println("the extrema of the end field is ", extrema(θ_F))
    println("the extrema of the ensemble average is ", extrema(θ̅_F))
    colormap = :bone_1
    # colormap = :nipy_spectral
    heatmap!(ax1, x_A, x_A, θ_A, colormap=colormap, colorrange=(0.0, 1.0), interpolate=true)
    hm = heatmap!(ax2, x_A, x_A, θ_F, colormap=colormap, colorrange=(0.0, 1.0), interpolate=true)
    hm_e = heatmap!(ax3, x_A, x_A, θ̅_F, colormap=colormap, colorrange=(0.0, 0.4), interpolate=true)
    Colorbar(fig[1, 3], hm, height=Relative(3 / 4), width=25, ticklabelsize=30, labelsize=30, ticksize=25, tickalign=1,)
    Colorbar(fig[1, 5], hm_e, height=Relative(3 / 4), width=25, ticklabelsize=30, labelsize=30, ticksize=25, tickalign=1,)
    display(fig)
end
