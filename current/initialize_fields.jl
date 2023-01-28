using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, HDF5, ProgressBars, Statistics
using CUDA
arraytype = CuArray

include("timestepping.jl")

rng = MersenneTwister(12345)
Random.seed!(12)

phase_speed = 1.0
N = 2^7
N_ens = 2^3 # 2^7
Ns = (N, N, N_ens)
κ = 1e-3 
ν = sqrt(1e-5 / 2) # raised to the hypoviscocity_power
ν_h = sqrt(1e-3) # raised to the dissipation_power
f_amp = 300
forcing_amplitude = f_amp * (N / 2^7)^2 # due to FFT nonsense
ϵ = 0.0 # large scale parameter, 0 means off, 1 means on
ω = 0.0 # frequency, 0 means no time dependence
t = [0.0]

Ω = S¹(4π)^2 × S¹(1)
grid = FourierGrid(Ns, Ω, arraytype=arraytype)
nodes, wavenumbers = grid.nodes, grid.wavenumbers

# build filter 
x = nodes[1]
y = nodes[2]
kˣ = wavenumbers[1]
kʸ = wavenumbers[2]
# construct waver
kxmax = maximum(kˣ)
kymax = maximum(kˣ)
kxmax = kymax = 30
waver = @. (kˣ)^2 + (kʸ)^2 ≤ ((kxmax / 2)^2 + (kymax / 2)^2)
waver = @. abs(kˣ) .+ 0 * abs(kʸ) ≤ 2 / 3 * kxmax
@. waver = waver * (0 * abs(kˣ) .+ 1 * abs(kʸ) ≤ 2 / 3 * kxmax)
waver[1, 1] = 0.0
waver[:, floor(Int, N / 2)+1] .= 0.0
waver[floor(Int, N / 2)+1, :] .= 0.0

# construct fields 
φ = arraytype(zeros(Ns...))
rand!(rng, φ)
φ *= 2π

field = arraytype(zeros(N, N))

##
# Fields 
# velocity
ψ = arraytype(zeros(ComplexF64, Ns...))
u = similar(ψ)
v = similar(ψ)
u₀ = similar(ψ)
v₀ = similar(ψ)

# prognostic variables
S = arraytype(zeros(ComplexF64, size(ψ)..., 2))
Ṡ = arraytype(zeros(ComplexF64, size(ψ)..., 2))

# auxiliary fields
uθ = similar(ψ)
vθ = similar(ψ)
uζ = similar(ψ)
vζ = similar(ψ)


∂ˣθ = similar(ψ)
∂ʸθ = similar(ψ)
∂ˣζ = similar(ψ)
∂ʸζ = similar(ψ)

∂ˣuθ = similar(ψ)
∂ʸvθ = similar(ψ)
∂ˣuζ = similar(ψ)
∂ʸvζ = similar(ψ)

𝒟θ = similar(ψ)
𝒟ζ = similar(ψ)
θ̇ = similar(ψ)

k₁ = similar(S)
k₂ = similar(S)
k₃ = similar(S)
k₄ = similar(S)
S̃ = similar(S)

# source
sθ = similar(ψ)
sζ = similar(ψ)

# phase
φ̇ = similar(φ)

# operators
∂x = im * kˣ
∂y = im * kʸ
Δ = @. ∂x^2 + ∂y^2

# plan ffts
P = plan_fft!(ψ, (1, 2))
P⁻¹ = plan_ifft!(ψ, (1, 2))

# time stepping
Δx = x[2] - x[1]
Δt = 1 / N

hypoviscocity_power = 2
dissipation_power = 2
# Dissipation 
Δ = @. ∂x^2 + ∂y^2
Δ⁻¹ = 1 ./ Δ
bools = (!).(isnan.(Δ⁻¹))
Δ⁻¹ .*= bools # hack in the fact that false * NaN = 0

𝒟ν = @. -(-ν_h * Δ⁻¹)^(hypoviscocity_power) - (-ν * Δ)^(dissipation_power) 
𝒟κ = @. κ * Δ

##
θ = view(S, :, :, :, 1)
ζ = view(S, :, :, :, 2)
@. θ = 0.0 * sin(3 * x) * sin(3 * y) + -0.1 + 0im
@. ζ = sin(3 * x) * sin(3 * y)

operators = (; P, P⁻¹, Δ⁻¹, waver, 𝒟ν, 𝒟κ, ∂x, ∂y)
auxiliary = (; ψ, x, y, φ, u, v, uζ, vζ, uθ, vθ, ∂ˣζ, ∂ʸζ, ∂ˣθ, ∂ʸθ, ∂ˣuζ, ∂ʸvζ, ∂ˣuθ, ∂ʸvθ, 𝒟θ, 𝒟ζ, sθ, sζ)
constants = (; forcing_amplitude=forcing_amplitude, ϵ=ϵ)
parameters = (; auxiliary, operators, constants)

function load_psi!(ψ; filename="/storage5/NonlocalPassiveTracers/Current/" * "proto_default_case.hdf5")
    fid = h5open(filename, "r")
    ψ .= arraytype(read(fid["stream function"]))
    close(fid)
    return nothing
end


#=
if initialize
    filename = "initial_streamfunction.hdf5"
    fid = h5open(filename, "w")
    P⁻¹ * ψ
    fid["psi"] = Array(ψ)
    close(fid)
end
=#
#=
if initialize
    using GLMakie
    fig2 = Figure()

    ax = Axis(fig2[1, 1])
    A_ζ = Array(real.(ζ))
    tmp = quantile(abs.(extrema(A_ζ))[:], 0.1)
    ax2 = Axis(fig2[1, 2])
    A_θ = Array(real.(θ))
    tmp2 = quantile(abs.(extrema(A_θ))[:], 0.1)

    sl_x = Slider(fig2[2, 1:2], range=1:N_ens, startvalue=1)
    o_index = sl_x.value

    field = @lift Array(real.(ζ[:, :, $o_index]))
    heatmap!(ax, field, colormap=:balance, colorrange=(-tmp, tmp), interpolate=false)

    field2 = @lift Array(real.(θ[:, :, $o_index]))
    heatmap!(ax2, field2, colormap=:balance, colorrange=(-tmp2, tmp2), interpolate=false)
    display(fig2)
end
=#