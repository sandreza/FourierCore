@info "initializing fields"
using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, HDF5, ProgressBars, Statistics
using CUDA
arraytype = CuArray
#=
N, N_ens, Ns = (N, N, N_ens)
are defined outside this script
=#

rng = MersenneTwister(12345)
Random.seed!(12)

phase_speed = 5.0

Ω = S¹(4π)^2 × S¹(1)
grid = FourierGrid(Ns, Ω, arraytype=arraytype)
nodes, wavenumbers = grid.nodes, grid.wavenumbers

# build filter 
x = nodes[1]
y = nodes[2]
kˣ = wavenumbers[1]
kʸ = wavenumbers[2]

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

# view into prognostic variables θ and ζ
θ = view(S, :, :, :, 1)
ζ = view(S, :, :, :, 2)
@. θ = 0.0 * sin(3 * x) * sin(3 * y) + -0.1 + 0im
@. ζ = sin(3 * x) * sin(3 * y)

auxiliary = (; ψ, x, y, φ, u, v, uζ, vζ, uθ, vθ, ∂ˣζ, ∂ʸζ, ∂ˣθ, ∂ʸθ, ∂ˣuζ, ∂ʸvζ, ∂ˣuθ, ∂ʸvθ, 𝒟θ, 𝒟ζ, sθ, sζ)

function load_psi!(ψ; filename= "case_1", directory = "/storage5/NonlocalPassiveTracers/Current/" )
    fid = h5open(directory * filename * ".hdf5", "r")
    ψ .= arraytype(read(fid["stream function"]))
    close(fid)
    return nothing
end

@info "done initializing fields"
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