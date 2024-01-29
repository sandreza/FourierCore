using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2, HDF5
using GLMakie, ProgressBars
# using CUDA

include("interpolation.jl")
Random.seed!(123456789)

arraytype = Array
L = 2π # 34 # 20π * sqrt(2) # 22
Ω = S¹(L)
N = 2^6  # number of gridpoints
grid = FourierGrid(N, Ω, arraytype=arraytype)
nodes, wavenumbers = grid.nodes, grid.wavenumbers

x = nodes[1]
k = wavenumbers[1]

# operators
∂x = im * k
Δ = @. ∂x^2
Δ² = @. Δ^2
Δ⁻¹ = 1 ./ (0.01 * Δ .- 1)
Δ⁻¹[1] = 0.0


Δx = x[2] - x[1]
c = 1
Δt = 0.1 * Δx / c

# fields 
uⁿ⁺¹ = zeros(ComplexF64, N)
uⁿ = zeros(ComplexF64, N)
u² = zeros(ComplexF64, N)

P! = plan_fft!(uⁿ, flags=FFTW.MEASURE)
P⁻¹! = plan_ifft!(uⁿ, flags=FFTW.MEASURE)

# initial condition
uⁿ .= exp.(-10 * (x .- π).^2)

##
op = 0.001 * Δ .- 1
regularization = real.(- 1 ./ op)

fig = Figure() 
ax = Axis(fig[1,1])
scatter!(ax, real.(uⁿ))
for i in 1:300
    P! * uⁿ
    @. u² = uⁿ - Δt * c * ∂x * uⁿ
    @. uⁿ⁺¹ = 0.5 * (uⁿ + u² - Δt * c * ∂x * u²)
    P⁻¹! * uⁿ⁺¹
    uⁿ .= uⁿ⁺¹
end
scatter!(ax, real.(uⁿ))
display(fig)

ax = Axis(fig[1,2])
scatter!(ax, real.(uⁿ))
for i in 1:300
    P! * uⁿ
    @. u² = uⁿ - Δt * c * op * ∂x * uⁿ
    @. uⁿ⁺¹ = 0.5 * (uⁿ + u² - Δt * c * op *  ∂x * u²)
    P⁻¹! * uⁿ⁺¹
    uⁿ .= uⁿ⁺¹
end
scatter!(ax, real.(uⁿ))
display(fig)

##
