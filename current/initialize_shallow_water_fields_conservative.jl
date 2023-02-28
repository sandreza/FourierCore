@info "initializing shallow water fields"
using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, HDF5, ProgressBars, Statistics
using CUDA
ArrayType = Array
#=
N, N_ens, Ns = (N, N, N_ens)
are defined outside this script
=#
include("timestepping.jl")

rng = MersenneTwister(12345)
Random.seed!(12)

phase_speed = 1.0
Ns = (64 * 2, 1)
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
dhudt = view(Ṡ, :, :, 2)
dhθdt = view(Ṡ, :, :, 3)
h = view(S, :, :, 1)
hu = view(S, :, :, 2)
hθ = view(S, :, :, 3)
# Field 
u = ArrayType(zeros(ComplexF64, Ns...))
θ = copy(u)
hu² = copy(u)
huθ = copy(u)
∂ˣhu = copy(u)
𝒟h = copy(∂ˣhu)
∂ˣhu² = copy(∂ˣhu)
∂ˣhu = copy(u)
∂ˣu = copy(u)
∂ˣh = copy(u)
𝒟hu = copy(u)
∂ˣhuθ = copy(u)
∂ˣθ = copy(u)
𝒟hθ = copy(u)
shu = copy(u)
φ = ArrayType(zeros((1, Ns[2])))
φ̇ = copy(φ)

## Plan FFT 
P = plan_fft!(u, 1)
P⁻¹ = plan_ifft!(u, 1)

## Initial conditions
@. h = 1
@. hθ = 1

@info "done initializing fields"

##
c = 0.1
g = 1.0
U = 1.0
φ_speed = 0.0

Δx = x[2] - x[1]
cfl = 0.01
Δt = cfl * Δx / U

ν = 0.2 # 0.1 * Δx^2 / Δt
κ = 0.2 # 0.1 * Δx^2 / Δt
𝒟ν = @. ν * Δ
𝒟κ = @. κ * Δ

operators = (; P, P⁻¹, 𝒟ν, 𝒟κ, ∂x)
constants = (; φ_speed, U, c, g)
auxiliary = (; φ, ∂ˣhu, 𝒟h, ∂ˣhu², ∂ˣu, ∂ˣh, 𝒟hu, ∂ˣhuθ, ∂ˣθ, 𝒟hθ, shu, u, θ, hu², huθ, x)
parameters = (; operators, constants, auxiliary)
t = [0.0]

rhs_shallow_water!(Ṡ, S, t, parameters)
##

timesnapshots = Vector{Float64}[]
for i in ProgressBar(1:100000)
    step_shallow_water!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, t, parameters)
    if i % 10 == 0
        push!(timesnapshots, Array(real.(h)[:, 1]))
    end
end

##
fig = Figure()
ax = Axis(fig[1, 1])
sl_x = Slider(fig[2, 1], range=1:length(timesnapshots), startvalue=1)
o_index = sl_x.value
field = @lift timesnapshots[$o_index]
scatter!(ax, field)
ylims!(ax, (0.0, 3.0))
display(fig)