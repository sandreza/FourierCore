using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2
using HDF5
using GLMakie
rng = MersenneTwister(12345)

Random.seed!(12)
include("transform.jl")
include("random_phase_kernel.jl")
# using GLMakie
using CUDA
arraytype = CuArray
Ω = S¹(2π)^2
N = 2^5 # number of gridpoints
grid = FourierGrid(N, Ω, arraytype=arraytype)
nodes, wavenumbers = grid.nodes, grid.wavenumbers

x = nodes[1]
y = nodes[2]
kˣ = wavenumbers[1]
kʸ = wavenumbers[2]
# construct filter
kxmax = maximum(kˣ)
kymax = maximum(kˣ)
filter = @. (kˣ)^2 + (kʸ)^2 ≤ ((kxmax / 2)^2 + (kymax / 2)^2)
filter = @. abs(kˣ) .+ 0 * abs(kʸ) ≤ 2 / 3 * kxmax
@. filter = filter * (0 * abs(kˣ) .+ 1 * abs(kʸ) ≤ 2 / 3 * kxmax)

# now define the random field 
wavemax = 5
𝓀 = arraytype([-wavemax, 0.0, wavemax]) # arraytype(1.0 .* [-wavemax, -wavemax + 1, wavemax - 1, wavemax])# arraytype(collect(-wavemax:1:wavemax))
𝓀ˣ = reshape(𝓀, (length(𝓀), 1))
𝓀ʸ = reshape(𝓀, (1, length(𝓀)))
# A = @. 0.1 * (𝓀ˣ * 𝓀ˣ + 𝓀ʸ * 𝓀ʸ)^(-11 / 12)
A = @. 0.5 * (𝓀ˣ * 𝓀ˣ + 𝓀ʸ * 𝓀ʸ)^(0.01) # @. 1e-1 / (1 * 2 * wavemax^2) .* (𝓀ˣ * 𝓀ˣ + 𝓀ʸ * 𝓀ʸ)^(0.0) # ( 1 .+ (0 .* 𝓀ˣ) .* 𝓀ʸ) 
A[A.==Inf] .= 0.0
φ = arraytype(2π * rand(size(A)...))
field = arraytype(zeros(N, N))

##
# Fields 
# velocity
ψ = arraytype(zeros(ComplexF64, N, N))
u = similar(ψ)
v = similar(ψ)

# auxiliary fields
S = arraytype(zeros(ComplexF64, size(ψ)..., 2))
Ṡ = arraytype(zeros(ComplexF64, size(ψ)..., 2))

∂ˣθ = similar(ψ)
∂ʸθ = similar(ψ)
∂ˣζ = similar(ψ)
∂ʸζ = similar(ψ)

𝒟θ = similar(ψ)
𝒟ζ = similar(ψ)
θ̇ = similar(ψ)

k₁ = similar(S)
k₂ = similar(S)
k₃ = similar(S)
k₄ = similar(S)
S̃ = similar(S)

# source
sθ = similar(ψ)
sζ = similar(ψ)

# phase
φ̇ = similar(A)

# operators
∂x = im * kˣ
∂y = im * kʸ
Δ = @. ∂x^2 + ∂y^2
Δ⁻¹ = 1 ./ Δ
bools = (!).(isnan.(Δ⁻¹))
Δ⁻¹ .*= bools # hack in the fact that false * NaN = 0

# plan ffts
P = plan_fft!(ψ)
P⁻¹ = plan_ifft!(ψ)

##
Δx = x[2] - x[1]
Δt = Δx / (2π) * 1
κ = 1.0 * Δx^2

# Dissipation 
𝒟 = @. κ * Δ - 1e-1 * (κ * Δ)^2 + 1e-3 * (κ * Δ)^3 - 1e-5 * (κ * Δ)^4

# take the initial condition as negative of the source
tic = Base.time()

# save some snapshots
ψ_save = typeof(real.(Array(ψ)))[]
θ_save = typeof(real.(Array(ψ)))[]

θ = view(S, :, :, 1)
ζ = view(S, :, :, 2)
@. θ = 0.0 * sin(3 * x) * sin(3 * y) + -0.1 + 0im
@. ζ = sin(3 * x) * sin(3 * y)

t = [0.0]
tend = 40 # 5000

phase_speed = 1.0

iend = ceil(Int, tend / Δt)

operators = (; P, P⁻¹, filter, Δ⁻¹, 𝒟, ∂x, ∂y)
auxiliary = (; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣζ, ∂ʸζ, ∂ˣθ, ∂ʸθ, 𝒟θ, 𝒟ζ, sθ, sζ)
constants = (; τ=4.0 * Δt, e=1e-2)# (; τ = 0.01, e = 0.01)

parameters = (; auxiliary, operators, constants)

function rhs!(Ṡ, S, parameters)
    θ̇ = view(Ṡ, :, :, 1)
    ζ̇ = view(Ṡ, :, :, 2)
    θ = view(S, :, :, 1)
    ζ = view(S, :, :, 2)


    P, P⁻¹, filter, Δ⁻¹, 𝒟, ∂x, ∂y = parameters.operators
    ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣζ, ∂ʸζ, ∂ˣθ, ∂ʸθ, 𝒟θ, 𝒟ζ, sθ, sζ = parameters.auxiliary
    τ, e = parameters.constants
    # construct random phase forcing
    event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
    wait(event)

    # construct source for vorticity
    @. sζ = ψ

    # P * ψ
    P * θ # in place fft ζ
    P * ζ # in place fft
    # grab stream function from vorticity
    @. ψ = Δ⁻¹ * ζ
    # ∇ᵖψ
    @. u = filter * (∂y * ψ)
    @. v = filter * -1.0 * (∂x * ψ)
    # ∇ζ
    @. ∂ˣθ = filter * ∂x * θ
    @. ∂ʸθ = filter * ∂y * θ
    @. ∂ˣζ = filter * ∂x * ζ
    @. ∂ʸζ = filter * ∂y * ζ
    @. 𝒟ζ = (Δ⁻¹ + 𝒟) * ζ
    @. 𝒟θ = 𝒟 * θ
    # go back to real space 
    P⁻¹ * u
    P⁻¹ * v

    P⁻¹ * ζ
    P⁻¹ * ∂ˣζ
    P⁻¹ * ∂ʸζ
    P⁻¹ * 𝒟ζ

    P⁻¹ * θ
    P⁻¹ * ∂ˣθ
    P⁻¹ * ∂ʸθ
    P⁻¹ * 𝒟θ

    # construct source 
    @. sθ = v + e - 1 / τ * θ * (real(θ) > 0)

    # rhs
    @. ζ̇ = real(-u * ∂ˣζ - v * ∂ʸζ + 𝒟ζ + sζ)
    @. θ̇ = real(-u * ∂ˣθ - v * ∂ʸθ + 𝒟θ + sθ)
    @. S = real(S)
    @. Ṡ = real(Ṡ)

    return nothing
end