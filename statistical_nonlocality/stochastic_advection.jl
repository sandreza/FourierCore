@info "initializing stochastic advection fields"
using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, HDF5, ProgressBars, Statistics
using CUDA
ArrayType = Array

include("timestepping.jl")

rng = MersenneTwister(12345)
Random.seed!(12)

Ns = (16, 1024)
Ω = S¹(2π) × S¹(1)

grid = FourierGrid(Ns, Ω, arraytype=ArrayType)
nodes, wavenumbers = grid.nodes, grid.wavenumbers

# build operators
x = nodes[1]
kˣ = wavenumbers[1]
∂x = im * kˣ
Δ = @. ∂x^2
κ = 0.01
𝒟 = κ .* Δ

θ = ArrayType(zeros(ComplexF64, Ns...))
source = ones(Ns[1], 1) .* (Ns[1] / 2)
source[1] = 0 # no mean source
source[floor(Int, Ns[1]/2)+1] = 0 # no aliased source
u = ArrayType(zeros((1, Ns[2])))
auxiliary = [copy(θ) for i in 1:5]

## Plan FFT 
P = plan_fft!(θ, 1)
P⁻¹ = plan_ifft!(θ, 1)

##
Δx = x[2] - x[1]
umax = 1
cfl = 0.3
dt = cfl * minimum([Δx / umax, Δx^2 / κ])
timetracker = zeros(2)
timetracker[2] = 0.1
stochastic = (; noise_amplitude=1.0, ou_amplitude=1.0)
operators = (; ∂x, 𝒟)

stochastic_auxiliary = copy(u)

parameters = (; operators, source, stochastic)
step! = StochasticRungeKutta4(auxiliary, stochastic_auxiliary, advection_rhs!, ou_rhs!, parameters, timetracker)
us = Vector{Float64}[]
uθs = Vector{Float64}[]
θs = Vector{Float64}[]
rng = MersenneTwister(12345)
fluxstart = 10000
iterations = 3 * 10000
for i in ProgressBar(1:iterations)
    step!(θ, u, rng)
    if (i > fluxstart) & (i % 10 == 0)
        push!(us, u[:])
        uθ = -imag(mean(u .* θ, dims=2))[:]
        push!(uθs, uθ[:])
        push!(θs, real.(mean(θ, dims=2)[:]))
    end
end

flux = mean(uθs)  / Ns[1] * 2
ensemble_mean = mean(θs) / Ns[1] * 2

keff1 = @. (1 / ensemble_mean - κ * kˣ^2) / kˣ^2
keff2 = @. flux / (ensemble_mean * kˣ)
keff1 = keff1[2:floor(Int, Ns[1]/2)]
keff2 = keff2[2:floor(Int, Ns[1] / 2)]