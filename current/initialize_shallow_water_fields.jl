@info "initializing shallow water fields"
using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, HDF5, ProgressBars, Statistics
using CUDA
ArrayType = CuArray
#=
N, N_ens, Ns = (N, N, N_ens)
are defined outside this script
=#
include("timestepping.jl")

rng = MersenneTwister(12345)
Random.seed!(12)

Ns = (128, 1024*4)
Ω = S¹(2π) × S¹(1)
grid = FourierGrid(Ns, Ω, arraytype=ArrayType)
nodes, wavenumbers = grid.nodes, grid.wavenumbers

# build operators
x = nodes[1]
kˣ = wavenumbers[1]
∂x = im * kˣ
Δ = @. ∂x^2


# Tendencies and State
S = ArrayType(zeros(ComplexF64, Ns..., 3))
Ṡ = ArrayType(zeros(ComplexF64, Ns..., 3))
k₁ = copy(S)
k₂ = copy(S)
k₃ = copy(S)
k₄ = copy(S)
S̃ = copy(S)
dhdt = view(Ṡ, :, :, 1)
dudt = view(Ṡ, :, :, 2)
dθdt = view(Ṡ, :, :, 3)
h = view(S, :, :, 1)
u = view(S, :, :, 2)
θ = view(S, :, :, 3)
# Field 
θ2 = ArrayType(zeros(ComplexF64, Ns...))
hu = copy(θ2)
u² = copy(θ2)
uθ = copy(θ2)
∂ˣu = copy(θ2)
𝒟h = copy(θ2)
∂ˣu² = copy(θ2)
∂ˣhu = copy(θ2)
∂ˣu = copy(θ2)
∂ˣh = copy(θ2)
𝒟u = copy(θ2)
∂ˣuθ = copy(θ2)
∂ˣθ = copy(θ2)
𝒟θ = copy(θ2)
shu = copy(θ2)
φ = ArrayType(zeros((1, Ns[2])))
φ̇ = copy(φ)

## Plan FFT 
P = plan_fft!(u, 1)
P⁻¹ = plan_ifft!(u, 1)

## Initial conditions
@. h = 1
@. θ = 1
@. u = 0

@info "done initializing fields"