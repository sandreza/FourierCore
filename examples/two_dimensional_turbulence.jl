using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2
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

# Note that the first order derivative operators only return a smooth derivative
∂ˣ = PointWiseSpectralOperator(filter .* ∂x)
∂ʸ = PointWiseSpectralOperator(filter .* ∂y)
∇² = PointWiseSpectralOperator(Δ)
∇⁻² = PointWiseSpectralOperator(Δ⁻¹)
∇ᵞ = PointWiseSpectralOperator(Δᵞ)
F = PointWiseSpectralOperator(filter)


# Ics
ψ = @. sin(x) * sin(y) + 0im
ζ = @. -sin(x) * sin(y) + 0im
dζ = @. -sin(x) * sin(y) + 0im
q = @. sin(x) + 0im * y
u = @. sin(x) * cos(y) + 0im
v = @. -cos(x) * sin(y) + 0im

# Fourier Transforms
ℱ! = plan_fft!(ψ)
ℱ⁻¹! = plan_ifft!(ψ)

# Tests: Quick Check
norm(-∂ˣ(ψ) - v)
norm(∂ʸ(ψ) - u)

function rhs!(dζ, ζ, b)
    ψ .= ∇⁻²(ζ)
    dζ .= F(∂ʸ(ψ) .* ∂ˣ(ζ) - ∂ʸ(ζ) .* ∂ˣ(ψ)) + 0.001 .* ∇²(ζ)
    return nothing
end

function new_rhs!(dS, S, b)
    # unpack state
    ζ = view(S, :, :, 1)
    dζ = view(dS, :, :, 1)
    q = view(S, :, :, 2)
    dq = view(dS, :, :, 2)
    # compute rhs
    ψ .= ∇⁻²(ζ)
    dζ .= ∂ʸ(ψ) .* ∂ˣ(ζ) - ∂ʸ(ζ) .* ∂ˣ(ψ) + 0.01 .* ∇ᵞ(ζ) # .* ∇²(ζ)
    dq .= ∂ʸ(ψ) .* ∂ˣ(q) - ∂ʸ(q) .* ∂ˣ(ψ) + 0.01 .* ∇ᵞ(q) # .* ∇²(q)
    return nothing
end

# Create State array and unpack
S = zeros(ComplexF64, size(ψ)..., 2)
ζ = view(S, :, :, 1)
q = view(S, :, :, 2)

min_δ = min(x[2] - x[1], y[2] - y[1])
cfl = 0.05
wavespeed = 2.0
dt = cfl * min_δ / wavespeed
rhs!(dζ, ζ, 0)

# odesolver = LSRK144(rhs!, ζ, dt; t0=0)
odesolver = LSRK144(new_rhs!, S, dt; t0=0)
@. ζ = -sin(1.0 * x) * sin(y) + -sin(2.0 * x) * sin(y) + 0im
@. q = -sin(1.0 * x) + sin(0.0 * y) + 0im

timeseries = []
timeend = 5
iterations = floor(Int, timeend / dt)
for i in 1:iterations
    dostep!(S, odesolver)
    if i % 40 == 0
        push!(timeseries, Array(real.(copy(S))))
    end
    if i % 100 == 0
        println("done with iteration ", i)
        println("the extrema of vorticity are ", extrema(real.(ζ)))
        println("the extrema of the tracer are ", extrema(real.(q)))
    end
end

#=
subgrid_filter = @. (kˣ)^2 + (kʸ)^2 ≤ ((kxmax / 2)^2 + (kymax / 2)^2)
function the_a_function(N, kmax, kcut, s)
    return log(1 + 4pi / N) / (kmax - kcut)^s
end

function Fₑ(k; kcut=70, s=8, a=1)
    cutoffbool = k > kcut
    return (1 - cutoffbool) * 1 + cutoffbool * exp(-a * (k - kcut)^s)
end
=#

##
fig = Figure()
ax = Axis(fig[1, 1], title="vorticity")
ax2 = Axis(fig[1, 2], title="tracer")

timeslider = Slider(fig[2, 1:2], range=1:1:length(timeseries), startvalue=1)
timeindex = timeslider.value
field1 = @lift(timeseries[$timeindex][:, :, 1])
field2 = @lift(timeseries[$timeindex][:, :, 2])
heatmap!(ax, field1, colorrange=(-2, 2), colormap=:balance, interpolate=true)
heatmap!(ax2, field2, colorrange=(-1, 1), colormap=:balance, interpolate=true)
display(fig)