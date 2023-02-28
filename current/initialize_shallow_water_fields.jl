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

Ns = (64*2, 1)
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

##
c = 0.1
g = 1.0
U = 1.0
φ_speed = 1.0

Δx = x[2] - x[1]
cfl = 0.2
Δt = cfl * Δx / maximum([U, c, κ / Δx, ν / Δx])

ν = 0.2 # 0.1 * Δx^2 / Δt
κ = 0.2 # 0.1 * Δx^2 / Δt
𝒟ν = @. ν * Δ
𝒟κ = @. κ * Δ

operators = (; P, P⁻¹, 𝒟ν, 𝒟κ, ∂x)
constants = (; φ_speed, U, c, g)
auxiliary = (; φ, ∂ˣhu, 𝒟h, ∂ˣu², ∂ˣu, ∂ˣh, 𝒟u, ∂ˣuθ, ∂ˣθ, 𝒟θ, shu, u, θ, u², uθ, x)
parameters = (; operators, constants, auxiliary)
t = [0.0]

rhs_shallow_water!(Ṡ, S, t, parameters)
##
Tend = 1000 
iterations = floor(Int, Tend / Δt)
timesnapshots_u = Vector{Float64}[]
timesnapshots_h = Vector{Float64}[]
timesnapshots_θ = Vector{Float64}[]
for i in ProgressBar(1:iterations)
    step_shallow_water!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, t, parameters)
    if i % 10 == 0
        push!(timesnapshots_u, Array(real.(u)[:, 1]))
        push!(timesnapshots_h, Array(real.(h)[:, 1]))
        push!(timesnapshots_θ, Array(real.(θ)[:, 1]))
    end
end

##
fig = Figure()
ax11 = Axis(fig[1, 1]; title = "h")
ax21 = Axis(fig[2, 1]; title = "u")
ax31 = Axis(fig[3, 1]; title = "θ")
sl_x = Slider(fig[4, 1], range=1:length(timesnapshots_u), startvalue=1)
o_index = sl_x.value
field = @lift timesnapshots_h[$o_index]
field2 = @lift timesnapshots_u[$o_index]
field3 = @lift timesnapshots_θ[$o_index]
scatter!(ax11, field)
ylims!(ax11, (0.0, 4.0))
scatter!(ax21, field2)
ylims!(ax21, (-2.0, 2.0))
scatter!(ax31, field3)
ylims!(ax31, (0.0, 4.0))
display(fig)