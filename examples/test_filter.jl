using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2, HDF5
using GLMakie, ProgressBars
# using CUDA

include("interpolation.jl")
Random.seed!(123456789)

arraytype = Array
L = 2π # 34 # 20π * sqrt(2) # 22
Ω = S¹(L) × S¹(L)
N = 2^10 # number of gridpoints
grid = FourierGrid(N, Ω, arraytype=arraytype)
nodes, wavenumbers = grid.nodes, grid.wavenumbers

x = nodes[1]
y = nodes[2]
kx = wavenumbers[1]
ky = wavenumbers[2]

∂x = im * kx
∂y = im * ky
Δ = @. ∂x^2 + ∂y^2

# operators
##
κ = 1e-4
L = 1 .- κ .* ( Δ .^3)
Lh = sqrt.(real.(L))
Lh⁻¹ = 1 ./ Lh
##
fig = Figure()

extrema(field)
for i in 1:2, j in 1:2
    ax = Axis(fig[i,j])
    field = real.(ifft(Lh⁻¹ .* (N/2)^2 .* (randn(N, N) .+ im * randn(N, N)) ))
    heatmap!(ax,field, colormap = :balance, colorrange = (-4, 4))
end
display(fig)