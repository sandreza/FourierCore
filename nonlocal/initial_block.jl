using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2
# using GLMakie, HDF5
using ProgressBars
rng = MersenneTwister(1234)
Random.seed!(123456789)
# grab computational kernels: functions defined
include("random_phase_kernel.jl")
# initialize fields: variables and domain defined here
include("initialize_fields.jl")

amplitude_factor = 10.0 # normalized later
phase_speed = 1.0 # sqrt(1.0 * 0.04) # makes the decorrelation time 1ish

# set the amplitude for the velocity field
A .= default_A * amplitude_factor

# number of gridpoints in transition is about λ * N / 2
bump(x; λ=20 / N[1], width=π / 4) = 0.5 * (tanh((x + width / 2) / λ) - tanh((x - width / 2) / λ))

##  Set diffusivity and timestep
Δx = x[2] - x[1]
κ = 5e-3
cfl = 0.1
Δx = (x[2] - x[1])
advective_Δt = cfl * Δx / amplitude_factor * 0.5
diffusive_Δt = cfl * Δx^2 / κ
Δt = minimum([advective_Δt, diffusive_Δt])

# save some snapshots
ψ_save = typeof(real.(Array(ψ)))[]
θ_save = typeof(real.(Array(ψ)))[]

r_A = Array(@. sqrt((x - π)^2 + (y - π)^2))
θ_A = [bump(r_A[i, j]) for i in 1:N[1], j in 1:N[1]]
θ .= CuArray(θ_A)
θclims = extrema(Array(real.(θ))[:])
P * θ # in place fft
@. κΔθ = κ * Δ * θ
P⁻¹ * κΔθ # in place fft
s .= -κΔθ
P⁻¹ * θ # in place fft
θ̅ .= 0.0

t = [0.0]
tend = 1
iend = ceil(Int, tend / Δt)

simulation_parameters = (; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ)
size_of_A = size(A)

realizations = 1

θ̅_timeseries = CuArray(zeros(size(ψ)..., iend))
θ_timeseries = Array(zeros(size(ψ)..., iend))

rhs! = θ_rhs_symmetric!
for j in ProgressBar(1:realizations)
    # new realization of flow
    rand!(rng, φ) # between 0, 1
    φ .*= 2π # to make it a random phase
    event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
    wait(event)
    t[1] = 0

    θ .= CuArray(θ_A)
    for i = 1:iend
        # fourth order runge_kutta
        rhs!(k₁, θ, simulation_parameters)
        @. θ̃ = θ + Δt * k₁ * 0.5
        φ_rhs_normal!(φ̇, φ, rng)
        @. φ += phase_speed * sqrt(Δt / 2) * φ̇
        rhs!(k₂, θ̃, simulation_parameters)
        @. θ̃ = θ + Δt * k₂ * 0.5
        rhs!(k₃, θ̃, simulation_parameters)
        @. θ̃ = θ + Δt * k₃
        φ_rhs_normal!(φ̇, φ, rng)
        @. φ += phase_speed * sqrt(Δt / 2) * φ̇
        rhs!(k₄, θ̃, simulation_parameters)
        @. θ += Δt / 6 * real(k₁ + 2 * k₂ + 2 * k₃ + k₄)
        @. θ = real(θ)

        t[1] += Δt
        @. θ̅_timeseries[:, :, i] += real.(θ) / realizations
        if j == 10
            θ_timeseries[:, :, i] = Array(real.(θ))
        end
    end

    @. θ̅ += θ / realizations
end

x_A = Array(x)[:] .- 2π
θ_F = Array(real.(θ))
θ̅_F = Array(real.(θ̅))

begin
    fig = Figure(resolution=(2048, 512))
    ax1 = Axis(fig[1, 1], title="t = 0")
    ax2 = Axis(fig[1, 2], title="instantaneous t = " * string(tend))
    ax3 = Axis(fig[1, 4], title="ensemble average t = " * string(tend))
    println("the extrema of the end field is ", extrema(θ_F))
    println("the extrema of the ensemble average is ", extrema(θ̅_F))
    colormap = :bone_1
    # colormap = :nipy_spectral
    heatmap!(ax1, x_A, x_A, θ_A, colormap=colormap, colorrange=(0.0, 0.4), interpolate=true)
    hm = heatmap!(ax2, x_A, x_A, θ_F, colormap=colormap, colorrange=(0.0, 0.4), interpolate=true)
    hm_e = heatmap!(ax3, x_A, x_A, θ̅_F, colormap=colormap, colorrange=(0.0, 0.4), interpolate=true)
    Colorbar(fig[1, 3], hm, height=Relative(3 / 4), width=25, ticklabelsize=30, labelsize=30, ticksize=25, tickalign=1,)
    Colorbar(fig[1, 5], hm_e, height=Relative(3 / 4), width=25, ticklabelsize=30, labelsize=30, ticksize=25, tickalign=1,)
    display(fig)
end
