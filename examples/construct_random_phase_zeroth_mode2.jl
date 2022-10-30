using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2
# GLMakie
using HDF5
using ProgressBars
rng = MersenneTwister(1234)
Random.seed!(123456789)
# jld_name = "high_order_timestep_spatial_tracer_"
jld_name = "default_streamfunction_"
include("transform.jl")
include("random_phase_kernel.jl")
# using GLMakie
save_fields = false
using CUDA
arraytype = CuArray
Ω = S¹(4π)^2
N = 2^7 # number of gridpoints
phase_speed = 1.0
amplitude_factor = 10.0


# for (di, amplitude_factor) in ProgressBar(enumerate([0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]))
di = 1
amplitude_factor = 1.0 # 0.5

grid = FourierGrid(N, Ω, arraytype=arraytype)
nodes, wavenumbers = grid.nodes, grid.wavenumbers

x = nodes[1]
y = nodes[2]
kˣ = wavenumbers[1]
kʸ = wavenumbers[2]
# construct filter
kxmax = maximum(kˣ)
kymax = maximum(kˣ)
filter = @. abs(kˣ) .+ 0 * abs(kʸ) ≤ 2 / 3 * kxmax
@. filter = filter * (0 * abs(kˣ) .+ 1 * abs(kʸ) ≤ 2 / 3 * kxmax)
filter = @. abs(kˣ) .+ 0 * abs(kʸ) ≤ Inf


# now define the random field 
wavemax = 3
𝓀 = arraytype(collect(-wavemax:0.5:wavemax))
𝓀ˣ = reshape(𝓀, (length(𝓀), 1))
𝓀ʸ = reshape(𝓀, (1, length(𝓀)))
A = @. (𝓀ˣ * 𝓀ˣ + 𝓀ʸ * 𝓀ʸ)^(-1)
A[A.==Inf] .= 0.0
φ = arraytype(2π * rand(size(A)...))
field = arraytype(zeros(N, N))

##
# Fields 
# velocity
ψ = arraytype(zeros(ComplexF64, N, N))
u = similar(ψ)
v = similar(ψ)

# theta
θ = similar(ψ)
∂ˣθ = similar(ψ)
∂ʸθ = similar(ψ)
κΔθ = similar(ψ)
θ̇ = similar(ψ)
s = similar(ψ)
θ̅ = similar(ψ)
k₁ = similar(ψ)
k₂ = similar(ψ)
k₃ = similar(ψ)
k₄ = similar(ψ)
θ̃ = similar(ψ)
uθ = similar(ψ)
vθ = similar(ψ)
∂ˣuθ = similar(ψ)
∂ʸvθ = similar(ψ)

# source
s = similar(ψ)
@. s = cos(kˣ[5] * x)

# phase
φ̇ = similar(A)

# operators
∂x = im * kˣ
∂y = im * kʸ
Δ = @. ∂x^2 + ∂y^2

# plan ffts
P = plan_fft!(ψ)
P⁻¹ = plan_ifft!(ψ)

##
φ .= 0.0
event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
wait(event)
P * ψ # in place fft
# ∇ᵖψ
@. u = -1.0 * (∂y * ψ)
@. v = (∂x * ψ)
# go back to real space 
P⁻¹ * ψ
P⁻¹ * θ
P⁻¹ * u
P⁻¹ * v
u₀ = sqrt(real(mean(u .* u)))
v₀ = sqrt(real(mean(v .* v)))
A .*= amplitude_factor * sqrt(2) / u₀
# check it 
event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
wait(event)
P * ψ # in place fft
# ∇ᵖψ
@. u = -1.0 * (∂y * ψ)
@. v = (∂x * ψ)
# go back to real space 
P⁻¹ * ψ
P⁻¹ * θ
P⁻¹ * u
P⁻¹ * v
u₀ = sqrt(real(mean(u .* u))) # / sqrt(2)
v₀ = sqrt(real(mean(v .* v))) # / sqrt(2)

##
# κ = 2 / N  # roughly 1/N for this flow
# κ = 2 / 2^8 # fixed diffusivity
# κ = 2e-4
Δx = x[2] - x[1]
κ = 0.01 * (2^7 / N)^2# amplitude_factor * 2 * Δx^2
cfl = 0.1
Δx = (x[2] - x[1])
advective_Δt = cfl * Δx / amplitude_factor
diffusive_Δt = cfl * Δx^2 / κ 
Δt = minimum([advective_Δt, diffusive_Δt]) 

# take the initial condition as negative of the source
maxind = minimum([40, floor(Int, N/4)])
index_choices = 2:maxind
tic = Base.time()

tstart = 1000
s .*= 0.0
for index_choice in ProgressBar(index_choices)
    kᶠ = kˣ[index_choice]
    @. θ = cos(kᶠ * x) / (kᶠ)^2 / κ # scaling so that source is order 1
    P * θ # in place fft
    @. κΔθ = κ * Δ * θ
    P⁻¹ * κΔθ # in place fft
    s .+= -κΔθ # add to source
end

t = [0.0]
tend = 5000 # 5000

iend = ceil(Int, tend / Δt)

# simulation_parameters = (; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, s, P, P⁻¹, filter)
simulation_parameters = (; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, filter) 

φ .= arraytype(2π * rand(size(A)...))
θ .*= 0.0
θ̅ .*= 0.0
size_of_A = size(A)

rhs! = θ_rhs_symmetric_zeroth!

for i = ProgressBar(1:iend)
    # fourth order runge kutta on deterministic part
    # keep ψ frozen is the correct way to do it here

    # the below assumes that φ is just a function of time
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
    @. θ += Δt / 6 * (k₁ + 2 * k₂ + 2 * k₃ + k₄)

    t[1] += Δt

    if t[1] >= tstart
        P * ψ # in place fft
        # ∇ᵖψ
        @. u = -1.0 * (∂y * ψ)
        P⁻¹ * u
        P⁻¹ * ψ # in place fft

        @. θ̅ += Δt * u * θ # direct accumation of fluxes, admittedly mislabed but w/e
    end

end

θ̅ ./= (t[end] - tstart)

effdiff = real(mean(θ̅))
