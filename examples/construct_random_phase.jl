using FourierCore, FourierCore.Grid, FourierCore.Domain
using GLMakie
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
𝓀 = collect(-3:0.5:3)
𝓀ˣ = reshape(𝓀, (length(𝓀), 1))
𝓀ʸ = reshape(𝓀, (1, length(𝓀)))
A = @. 0.1 * (𝓀ˣ * 𝓀ˣ + 𝓀ʸ * 𝓀ʸ)^(-11 / 12)
A[a .== Inf] .= 0.0
θ = 2π * rand(size(A)...)
field = zeros(N, N)

for i in eachindex(𝓀ˣ), j in eachindex(𝓀ʸ)
    @. field += A[i, j] * cos(𝓀ˣ[i] * x + 𝓀ʸ[j] * y + θ[i, j])
end
