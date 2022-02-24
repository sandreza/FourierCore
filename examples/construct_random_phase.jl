using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools
include("transform.jl")
# using GLMakie

Ω = S¹(4π)^2
N = 2^6 # number of gridpoints
Nϕ = 11 # number of random phases
@assert Nϕ < N
grid = FourierGrid(N, Ω)
(; nodes, wavenumbers) = grid
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
𝓀 = collect(-wavemax:0.5:wavemax)
𝓀ˣ = reshape(𝓀, (length(𝓀), 1))
𝓀ʸ = reshape(𝓀, (1, length(𝓀)))
A = @. 0.1 * (𝓀ˣ * 𝓀ˣ + 𝓀ʸ * 𝓀ʸ)^(-11 / 12)
A[A.==Inf] .= 0.0
φ = 2π * rand(size(A)...)
field = zeros(N, N)

# Expensive
function random_phase(field, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
    field .= 0.0
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
@. s = cos(kˣ[5] * x)

# phase
φ̇ = similar(A)

# operators
∂x = im * kˣ
∂y = im * kʸ
Δ = @. ∂x^2 + ∂y^2

# plan ffts
P = plan_fft!(ψ)
P⁻¹ = plan_ifft!(ψ)

##
κ = 1e-2 # 1e-4
Δt = (x[2] - x[1]) / 4π

ψ_save = typeof(real.(ψ))[]
θ_save = typeof(real.(ψ))[]

# take the initial condition as negative of the source
@. s = cos(kˣ[5] * x)
θ .= -s
s .= 0.0
tic = Base.time()
for i = 1:1000
    random_phase(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
    # spectral space representation 
    P * ψ # in place fft
    P * θ # in place fft
    # ∇ᵖψ
    @. u = filter * -1.0 * (∂y * ψ) 
    @. v = filter * (∂x * ψ) 
    # ∇θ
    @. ∂ˣθ = filter * ∂x * θ
    @. ∂ʸθ = filter * ∂y * θ
    @. κΔθ = κ * Δ * θ
    # go back to real space 
    P⁻¹ * ψ
    P⁻¹ * θ
    P⁻¹ * u
    P⁻¹ * v
    P⁻¹ * ∂ˣθ
    P⁻¹ * ∂ʸθ
    P⁻¹ * κΔθ
    # Assemble RHS
    φ̇ .= 2π * (rand(size(A)...) .- 1) * 0.01
    @. θ̇ = -u * ∂ˣθ - v * ∂ʸθ + κΔθ + s
    # Euler step
    @. φ += sqrt(Δt) * φ̇
    @. θ += Δt * θ̇

    # save output
    if i % 10 == 0
        push!(ψ_save, real.(ψ))
        push!(θ_save, real.(θ))
    end
end
toc = Base.time()
println("the time for the simiulation was ", toc - tic, " seconds")

##
using GLMakie

time_index = Observable(1)
ψfield = @lift(ψ_save[$time_index])
θfield = @lift(θ_save[$time_index])
fig = Figure(resolution = (1722, 1076))
ax = Axis(fig[1, 1]; title = "stream function ")
ax2 = Axis(fig[1, 2]; title = "tracer concentration")
heatmap!(ax, ψfield, interpolate = true, colormap = :balance, colorrange = (-1.5, 1.5))
heatmap!(ax2, θfield, interpolate = true, colormap = :balance, colorrange = (-1.0, 1.0))
display(fig)
for i in eachindex(ψ_save)
    sleep(0.1)
    time_index[] = i
end
