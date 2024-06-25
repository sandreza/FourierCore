@info "initializing fields"
using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, HDF5, ProgressBars, Statistics
arraytype = Array
N = 32
Ns = (N, N , 4)

rng = MersenneTwister(12345)
Random.seed!(12)

phase_speed = 1.0

Ω = S¹(2π)^2 × S¹(1)
grid = FourierGrid(Ns, Ω, arraytype=arraytype)
nodes, wavenumbers = grid.nodes, grid.wavenumbers

# build filter 
x = nodes[1]
y = nodes[2]
kˣ = wavenumbers[1]
kʸ = wavenumbers[2]

# construct fields 
φ = arraytype(zeros(Ns...))
rand!(rng, φ)
φ *= 2π

field = arraytype(zeros(N, N))

##
# Fields 
# velocity
ψ = arraytype(zeros(ComplexF64, Ns...))
u = similar(ψ)
v = similar(ψ)
u₀ = similar(ψ)
v₀ = similar(ψ)

# prognostic variables
S = arraytype(zeros(ComplexF64, size(ψ)..., 1))
Ṡ = arraytype(zeros(ComplexF64, size(ψ)..., 1))

# auxiliary fields
uθ = similar(ψ)
vθ = similar(ψ)
uζ = similar(ψ)
vζ = similar(ψ)

∂ˣθ = similar(ψ)
∂ʸθ = similar(ψ)
∂ˣζ = similar(ψ)
∂ʸζ = similar(ψ)

∂ˣuθ = similar(ψ)
∂ʸvθ = similar(ψ)
∂ˣuζ = similar(ψ)
∂ʸvζ = similar(ψ)

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
φ̇ = similar(φ)

# view into prognostic variables θ and ζ
θ = view(S, :, :, :, 1)
ζ = view(S, :, :, :, 1)
@. θ = 0.0 * sin(3 * x) * sin(3 * y) + -0.1 + 0im
@. ζ = sin(3 * x) * sin(3 * y)

auxiliary = (; ψ, x, y, φ, u, v, uζ, vζ, uθ, vθ, ∂ˣζ, ∂ʸζ, ∂ˣθ, ∂ʸθ, ∂ˣuζ, ∂ʸvζ, ∂ˣuθ, ∂ʸvθ, 𝒟θ, 𝒟ζ, sθ, sζ)

@info "done initializing fields"
