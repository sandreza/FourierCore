using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2
import FourierCore.OverEngineering: PointWiseSpectralOperator

include("ode.jl")

arraytype = Array
Ω = S¹(2π)^2
N = 2^6 # number of gridpoints
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

Δ = @. ∂x^2 + ∂y^2
Δ⁻¹ = 1 ./ Δ
bools = (!).(isnan.(Δ⁻¹))
Δ⁻¹ .*= bools # hack in the fact that false * NaN = 0

∂ˣ = PointWiseSpectralOperator(∂x)
∂ʸ = PointWiseSpectralOperator(∂y)
∇² = PointWiseSpectralOperator(Δ)
∇⁻² = PointWiseSpectralOperator(Δ⁻¹)
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
    ζ = view(S, :, :, 1)
    dζ = view(dS, :, :, 1)
    q = view(S, :, :, 2)
    dq = view(dS, :, :, 2)
    ψ .= ∇⁻²(ζ)
    dζ .= F(∂ʸ(ψ) .* ∂ˣ(ζ) - ∂ʸ(ζ) .* ∂ˣ(ψ)) + 0.001 .* ∇²(ζ)
    dq .= F(∂ʸ(ψ) .* ∂ˣ(q) - ∂ʸ(q) .* ∂ˣ(ψ)) + 0.001 .* ∇²(q)
    return nothing
end

# Create State array and unpack
S = zeros(ComplexF64, size(ψ)..., 2)
ζ = view(S, :, :, 1)
# q = view(S, :, :, 2)

min_δ = min(x[2] - x[1], y[2] - y[1])
cfl = 0.05
wavespeed = 2.0
dt = cfl * min_δ / wavespeed
rhs!(dζ, ζ, 0)

odesolver = LSRK144(rhs!, ζ, dt; t0=0)
@. ζ = -sin(1.0 * x) * sin(y) + -sin(2.0 * x) * sin(y) + 0im

timeseries = []
iterations = 200
for i in 1:iterations
    dostep!(ζ, odesolver)
    if i % 40 == 0
        push!(timeseries, Array(real.(copy(ζ))))
    end
    if i % 100 == 0
        println("done with iteration ", i)
        println("the extrema of vorticity are ", extrema(real.(ζ)))
    end
end

subgrid_filter = @. (kˣ)^2 + (kʸ)^2 ≤ ((kxmax / 2)^2 + (kymax / 2)^2)
# kcut = 70 * kmin  or 70 * kx[2]
# kmin / kforcing = 60
# r = 0.03

function the_a_function(N, kmax, kcut, s)
    return log(1 + 4pi / N) / (kmax - kcut)^s
end

function Fₑ(k; kcut=70, s=8, a=1)
    cutoffbool = k > kcut
    return (1 - cutoffbool) * 1 + cutoffbool * exp(-a * (k - kcut)^s)
end

##
fig = Figure()
ax = Axis(fig[1, 1])

timeslider = Slider(fig[2, 1], range=1:1:length(timeseries), startvalue=1)
timeindex = timeslider.value
field = @lift(timeseries[$timeindex])
heatmap!(ax, field, colorrange=(-2, 2), colormap=:balance, interpolate=true)
display(fig)