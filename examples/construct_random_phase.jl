using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools
include("transform.jl")
# using GLMakie

Ω = S¹(4π)^2
N = 2^6
Nϕ = 11 # number of random phases
@assert Nϕ < N
grid = FourierGrid(N, Ω)
(; nodes, wavenumbers) = grid
x = nodes[1]
y = nodes[2]
kˣ = wavenumbers[1]
kʸ = wavenumbers[2]

# now define the random field 
wavemax = 3
𝓀 = collect(-wavemax:0.5:wavemax)
𝓀ˣ = reshape(𝓀, (length(𝓀), 1))
𝓀ʸ = reshape(𝓀, (1, length(𝓀)))
A = @. 0.1 * (𝓀ˣ * 𝓀ˣ + 𝓀ʸ * 𝓀ʸ)^(-11 / 12)
A[A.==Inf] .= 0.0
φ = 2π * rand(size(A)...)
field = zeros(N, N)

function random_phase()
    for i in eachindex(𝓀ˣ), j in eachindex(𝓀ʸ)
        @. field += A[i, j] * cos(𝓀ˣ[i] * x + 𝓀ʸ[j] * y + φ[i, j])
    end
end

𝒯 = Transform(grid)
field1 = field .+ 0 * im
field2 = similar(field1)
mul!(field2, 𝒯.forward, field1)

# @benchmark mul!(field1, 𝒯.backward, field2)
# @benchmark mul!(field1, 𝒯.backward, field2)

##
# Fields 
# velocity
ψ = zeros(ComplexF64, N, N)
u = similar(ψ)
v = similar(ψ)

# theta
θ = similar(ψ)
∂ˣθ = similar(ψ)
∂ʸθ = similar(ψ)
κΔθ = similar(ψ)
θ̇ = similar(ψ)
s = similar(ψ)

# source
s = similar(ψ)

# phase
φ̇ = similar(A)

# operators
∂x = im * kˣ
∂y = im * kʸ
Δ = @. ∂x^2 + ∂y^2

# update 
@. ψ = sin(kˣ[2] * x) * cos(kʸ[2] * y)
ψ̂ = similar(ψ)
P = plan_fft!(ψ)
P⁻¹ = plan_ifft!(ψ)
# @benchmark P * ψ
mul!(ψ̂, 𝒯.forward, ψ)
ψ̂ .*= ∂x
tmp = 𝒯.backward * ψ̂

ℱ = 𝒯.forward
ℱ⁻¹ = 𝒯.backward

u .= -1.0 * ℱ⁻¹ * (∂y .* (ℱ * ψ))
v .= ℱ⁻¹ * (∂x .* (ℱ * ψ))

P * θ # in place fft
∂ˣθ .= ℱ⁻¹ * (∂x .* θ)
∂ʸθ .= ℱ⁻¹ * (∂y .* θ)

##
κ = 1e-4
Δt = 0.1
@benchmark begin
for i = 1:1
    P * ψ # in place fft
    u .= -1.0 * ℱ⁻¹ * (∂y .* ψ)
    v .= ℱ⁻¹ * (∂x .* ψ)
    P * θ # in place fft
    ∂ˣθ .= ℱ⁻¹ * (∂x .* θ)
    ∂ʸθ .= ℱ⁻¹ * (∂y .* θ)
    κΔθ .= κ * Δ * θ
    # Assemble RHS
    φ̇ .= 2π * rand(size(A)...)
    @. θ̇ = u * ∂ˣθ + v * ∂ʸθ + κΔθ + s
    # Euler step
    @. φ += sqrt(Δt) * φ̇
    @. θ += Δt * θ̇
end
end

