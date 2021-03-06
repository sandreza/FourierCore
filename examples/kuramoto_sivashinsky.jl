using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2, HDF5
using GLMakie
# using CUDA

Random.seed!(123456789)


arraytype = Array
L = 22 # 20π * sqrt(2) # 22
Ω = S¹(L)
N = 2^5 * 8  # number of gridpoints
grid = FourierGrid(N, Ω, arraytype=arraytype)
nodes, wavenumbers = grid.nodes, grid.wavenumbers

x = nodes[1]
k = wavenumbers[1]
# construct filter
kmax = maximum(k)

filter = @. abs(k) ≤ 0.5 * kmax # 2 / 3 * kmax

# operators
∂x = im * k
Δ = @. ∂x^2
Δ² = @. Δ^2

Δx = x[2] - x[1]
c = 3
Δt = 0.3 * Δx / c
inv_op = @. 1 / (1 + Δt * Δ + Δt * Δ²)

# fields 
uⁿ⁺¹ = zeros(ComplexF64, N)
uⁿ = zeros(ComplexF64, N)
u²x = zeros(ComplexF64, N)
u² = zeros(ComplexF64, N)

P! = plan_fft!(uⁿ)
P⁻¹! = plan_ifft!(uⁿ)


# timestepping

function nonlinear_term!(u²x, uⁿ, u², P!, P⁻¹!, ∂x, filter)
    # always assume u is in fourier space
    # use 2/3 rule for stability
    @. u² = filter * uⁿ
    P⁻¹! * u²
    @. u² = u² * u²
    P! * u²
    @. u²x = -0.5 * ∂x * u²

    return nothing
end

@. uⁿ = sin(2π / L * x) + 0.1 * cos(2 * 2π / L * x) + 0.1 * sin(11 * 2π / L * x)
P! * uⁿ

substep = 30
endtime = 1000
M = floor(Int, endtime / (substep * Δt))
plotmat = zeros(N, M)

tic = Base.time()
for i in 1:M
    for j in 1:substep
        nonlinear_term!(u²x, uⁿ, u², P!, P⁻¹!, ∂x, filter)
        @. uⁿ⁺¹ = inv_op * (uⁿ + Δt * u²x)
        @. uⁿ = uⁿ⁺¹
    end
    P⁻¹! * uⁿ
    @. uⁿ = real(uⁿ)
    plotmat[:, i] .= real.(uⁿ)
    P! * uⁿ
end
toc = Base.time()
println("The time for the simulation was ", toc - tic, " seconds")

fig = Figure(resolution=(1000, 500))
ax = Axis(fig[1, 1])
ax_spec = Axis(fig[1, 2])
ax_inst = Axis(fig[1, 3])
contourf!(ax, plotmat)
scatter!(ax_spec, log.(abs.(uⁿ) .+ eps(1.0)))
lines!(ax_inst, plotmat[:, end])
display(fig)


if N == 32
    fid = h5open("low_res.h5", "w")
    println("saving low rez")
    fid["u"] = plotmat
    close(fid)
elseif N == 256
    fid = h5open("high_res.h5", "w")
    println("saving high rez")
    fid["u"] = plotmat
    close(fid)
end