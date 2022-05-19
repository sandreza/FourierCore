using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2
using KernelAbstractions
# using CUDA
import FourierCore.OverEngineering: PointWiseSpectralOperator

include("ode.jl")

arraytype = Array
Ω = S¹(2π)^2
N = 2^7 # number of gridpoints
grid = FourierGrid(N, Ω, arraytype=arraytype)
(; nodes, wavenumbers) = grid

x = nodes[1]
y = nodes[2]
kˣ = wavenumbers[1]
kʸ = wavenumbers[2]

# construct filter
kxmax = maximum(kˣ)
kymax = maximum(kˣ)
filter = @. (kˣ)^2 + (kʸ)^2 ≤ ((kxmax / 2)^2 + (kymax / 2)^2)


# Operators
∂x = im * kˣ
∂y = im * kʸ

# to convert things to damping timescale of smallest wavelength
δ = min(x[2] - x[1], y[2] - y[1])

Δ = @. ∂x^2 + ∂y^2
Δᵞ = @. δ^2 * Δ - δ^4 * Δ^2 + δ^6 * Δ^3 - δ^8 * Δ^4 # δ^2 * Δ * exp(-2 * δ^2 * Δ) 
# Δᵞ = @. δ^2 * Δ * exp(-δ^2 * Δ)

Δ⁻¹ = 1 ./ Δ
bools = (!).(isnan.(Δ⁻¹))
Δ⁻¹ .*= bools # hack in the fact that false * NaN = 0

kmax = maximum(kˣ)
kcutoff = floor(Int, kmax / 2)
s = 1
a = log(1 + 4π / N) / (kmax - kcutoff)^s
k = @. sqrt((kˣ)^2 + (kʸ)^2)
filter_mask = @. k ≤ kcutoff
expfilter = @. exp(-a * (k - kcutoff)^s) * (1 - filter_mask) + filter_mask

# Note that the first order derivative operators only return a smooth derivative
∂ˣ = PointWiseSpectralOperator(filter .* ∂x)
∂ʸ = PointWiseSpectralOperator(filter .* ∂y)
∇² = PointWiseSpectralOperator(Δ)
∇⁻² = PointWiseSpectralOperator(Δ⁻¹)
∇ᵞ = PointWiseSpectralOperator(Δᵞ)
Filter = PointWiseSpectralOperator(expfilter)

# Ics
ψ = @. sin(x) * sin(y) + 0im
ζ = @. -sin(x) * sin(y) + 0im
dζ = @. -sin(x) * sin(y) + 0im
q = @. sin(x) + 0im * y
u = @. sin(x) * cos(y) + 0im
v = @. -cos(x) * sin(y) + 0im

# Tests: Quick Check
norm(-∂ˣ(ψ) - v)
norm(∂ʸ(ψ) - u)

#=
random forcing prep
rng = MersenneTwister(1234)
# Random.seed!(123456789)
Random.seed!(12)
φ̇ = arraytype(zeros(12,12))
rng = MersenneTwister(1234)
randn!(rng, φ̇) 
φ̇ .*= sqrt(1 / 12)
=#


function rhs!(dS, S, b)
    # unpack state
    ζ = view(S, :, :, 1)
    dζ = view(dS, :, :, 1)
    # q = view(S, :, :, 2)
    # dq = view(dS, :, :, 2)
    q₁ = view(S, :, :, 2)
    dq₁ = view(dS, :, :, 2)

    q₁ .= Filter(q₁)
    ζ .= Filter(ζ)

    τ = 1e-2
    γ = 1.0
    e = 0.1
    c = 1 / τ * q₁ .* (real.(q₁) .> 0)
    # compute rhs
    ψ .= ∇⁻²(ζ)
    dζ .= ∂ʸ(ψ) .* ∂ˣ(ζ) - ∂ʸ(ζ) .* ∂ˣ(ψ) - 0.03 * ζ + 0.00 .* ∇ᵞ(ζ)
    # should evolve q₁ = q - qₛ where qₛ = γ * y # q = ⟨q⟩ + q'
    # q = q₁ + qₛ
    # dq .= ∂ʸ(ψ) .* ∂ˣ(q) - ∂ʸ(q) .* ∂ˣ(ψ) + 0.01 .* ∇ᵞ(q)
    dq₁ .= ∂ʸ(ψ) .* ∂ˣ(q₁) - ∂ʸ(q₁) .* ∂ˣ(ψ) - γ * ∂ˣ(ψ) + 0.00 .* ∇ᵞ(q₁)
    dq₁ .+= e .- c

    dq₁ .= Filter(dq₁)
    dζ .= Filter(dζ)
    return nothing
end

function stochastic_update!(S, dt, x, y)
    ζ = view(S, :, :, 1)
    a = 1.0
    a1 = a * 0.3 * randn()
    a2 = a * 0.3 * randn()
    a3 = a * 0.3 * randn()
    a4 = a * 0.3 * randn()
    a5 = a * 0.3 * randn()
    a6 = a * 0.3 * randn()
    a7 = a * 0.3 * randn()
    a8 = a * 0.3 * randn()
    a9 = a * 0.3 * randn()
    forcing = @. a1 * cos(2 * x) * cos(2 * y) + a2 * cos(3 * x) * cos(3 * y) + a5 * cos(2 * x) * cos(3 * y) + a6 * cos(3 * x) * cos(2 * y)
    forcing = @. a3 * sin(1 * x) * sin(3 * y) + a4 * sin(3 * x) * sin(1 * y)
    forcing = @. a7 * sin(2 * x) * cos(2 * y) + a8 * cos(2 * x) * sin(2 * y) + a9 * sin(2 * x) * sin(2 * y)
    ζ .+= sqrt(dt) .* forcing
end

# Legit forcing from Shafer Smith! Not that legit of a description though!!!
# c = 0.99
# theta = 2\pi * rand()
# F̂\_n = c F_n−1 + (1 − c\^2)\^(1/2) * exp(i * \theta)
# F\_n = havg(\psi F̂\_n)\^-1 F̂\_n
# havg here denotes the horizontal average across the domain.


# Create State array and unpack
S = arraytype(zeros(ComplexF64, size(ψ)..., 2))
ζ = view(S, :, :, 1)
q = view(S, :, :, 2)

min_δ = min(x[2] - x[1], y[2] - y[1])
cfl = 0.05
wavespeed = 2.0
dt = cfl * min_δ / wavespeed
# rhs!(dζ, ζ, 0)

# odesolver = LSRK144(rhs!, ζ, dt; t0=0)
odesolver = LSRK144(rhs!, S, dt; t0=0)
@. ζ = -sin(1.0 * x) * sin(y) + -sin(2.0 * x) * sin(y) + 0im
@. q = -sin(1.0 * x) + sin(0.0 * y) + 0im

timeseries = []
timeend = 10
iterations = floor(Int, timeend / dt)
for i in 1:iterations
    dostep!(S, odesolver)
    stochastic_update!(S, dt, x, y)
    if i % 40 == 0
        push!(timeseries, Array(real.(copy(S))))
    end
    if i % 100 == 0
        println("done with iteration ", i)
        println("the extrema of vorticity are ", extrema(real.(ζ)))
        println("the extrema of the tracer are ", extrema(real.(q)))
    end
end

##
using GLMakie
fig = Figure()
ax = Axis(fig[1, 1], title="vorticity")
ax2 = Axis(fig[1, 3], title="saturation")

timeslider = Slider(fig[2, 1:2], range=1:1:length(timeseries), startvalue=1)
timeindex = timeslider.value
field1 = @lift(timeseries[$timeindex][:, :, 1])
field2 = @lift(timeseries[$timeindex][:, :, 2])
hm_ζ = heatmap!(ax, field1, colorrange=(-1, 1), colormap=:balance, interpolate=true)
Colorbar(fig[1, 2], hm_ζ, height=Relative(3 / 4), width=25, ticklabelsize=30, labelsize=30, ticksize=25, tickalign=1,)

hm_q = heatmap!(ax2, field2, colorrange=(-5, 0.02), colormap=:solar, interpolate=true)
Colorbar(fig[1, 4], hm_q, height=Relative(3 / 4), width=25, ticklabelsize=30, labelsize=30, ticksize=25, tickalign=1,)
display(fig)