using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2
rng = MersenneTwister(1234)
Random.seed!(123456789)

include("transform.jl")
include("random_phase_kernel.jl")

using CUDA
arraytype = CuArray
Ω = S¹(4π)^2
N = 2^9 # number of gridpoints
grid = FourierGrid(N, Ω, arraytype = arraytype)
nodes, wavenumbers = grid.nodes, grid.wavenumbers

x = nodes[1]
y = nodes[2]
kˣ = wavenumbers[1]
kʸ = wavenumbers[2]
# construct filter
kxmax = maximum(kˣ)
kymax = maximum(kˣ)
filter = @. (kˣ)^2 + (kʸ)^2 ≤ ((kxmax / 2)^2 + (kymax / 2)^2)

# now define the random field 
wavemax = 3
𝓀 = arraytype(collect(-wavemax:0.5:wavemax))
𝓀ˣ = reshape(𝓀, (length(𝓀), 1))
𝓀ʸ = reshape(𝓀, (1, length(𝓀)))
A = @. 0.1 * (𝓀ˣ * 𝓀ˣ + 𝓀ʸ * 𝓀ʸ)^(-11 / 12)
A[A.==Inf] .= 0.0
φ = arraytype(2π * rand(size(A)...))
field = arraytype(zeros(N, N))
φ̇ = copy(φ)
p = (; rng)


function ode_φ(dφ, φ, p, t)
    dφ .= 0.0
    return nothing
end

function σ_φ(dφ, φ, p, t)
    dφ .= 1 / sqrt(12)
    return nothing
end

Δt = (x[2] - x[1]) / (4π)
prob_φ_lorenz = SDEProblem(ode_φ, σ_φ, φ, (0.0, 10.0))
φ_init = copy(φ)
sol = solve(prob_φ_lorenz, SOSRI(), save_everystep = false, save_end = false, dt = Δt, adaptive = false)

@benchmark begin
    for i in 1:5120
        @. φ += sqrt(Δt) * φ̇
        φ_rhs!(φ̇, φ, rng)
    end
end

