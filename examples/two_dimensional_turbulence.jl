using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2

arraytype = Array
Ω = S¹(2π)^2
N = 2^4 # number of gridpoints
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

# Ics
ψ = @. sin(x) * sin(y) + 0im
ζ = @. -sin(x) * sin(y) + 0im
q = @. sin(x) + 0im * y
u = @. sin(x) * cos(y) + 0im
v = @. -cos(x) * sin(y) + 0im

# Fourier Transforms
ℱ! = plan_fft!(ψ)
ℱ⁻¹! = plan_ifft!(ψ)

# Tests: Quick Check
ℱ! * ψ
ψˣ = ∂x .* ψ
ℱ⁻¹! * ψˣ
ℱ⁻¹! * ψ
norm(-ψˣ - v)

ℱ! * ψ
ψʸ = ∂y .* ψ
ℱ⁻¹! * ψʸ
ℱ⁻¹! * ψ
norm(ψʸ - u)

# ∂ₜζ + J(ψ, ζ) = F
# ∂ₜq + J(ψ, q) = stuff
function advection(ζ, q)
    ℱ! * ζ # in place
    ℱ! * q # in place
    
    ψ = Δ⁻¹ .* ζ 
    ψˣ = filter .* ∂x .* ψ
    ψʸ = filter .* ∂y .* ψ
    ζˣ = filter .* ∂x .* ζ
    ζʸ = filter .* ∂y .* ζ
    qˣ = filter .* ∂x .* q
    qʸ = filter .* ∂y .* q

    ℱ⁻¹! * ζ # in place 
    ℱ⁻¹! * q # in place 
    ℱ⁻¹! * ψˣ # in place
    ℱ⁻¹! * ζˣ # in place 
    ℱ⁻¹! * ψʸ # in place
    ℱ⁻¹! * ζʸ # in place 

    ζ̇ = @. ψʸ * ζˣ - ψˣ * ζʸ 
    q̇ = @. ψʸ * qˣ - ψˣ * qʸ 
    return (; ζ̇, q̇)
end

(; ζ̇, q̇) = advection(ζ, q)

subgrid_filter = @. (kˣ)^2 + (kʸ)^2 ≤ ((kxmax / 2)^2 + (kymax / 2)^2)

# kcut = 70 * kmin  or 70 * kx[2]
# kmin / kforcing = 60
# r = 0.03
function the_a_function(N, kmax, kcut, s)
    return log(1 + 4pi / N) / (kmax - kcut)^s
end
function Fₑ(k; kcut = 70, s = 8, a = 1)
    cutoffbool = k > kcut
    return (1 - cutoffbool) * 1 + cutoffbool * exp(-a * (k - kcut)^s)
end

Fₑ( 72)

q̇ .+= forcing
ζ̇ .+= forcing

