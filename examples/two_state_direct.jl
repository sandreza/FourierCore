using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2
# using GLMakie, HDF5
using ProgressBars
rng = MersenneTwister(1234)
Random.seed!(123456789)

include("transform.jl")
include("random_phase_kernel.jl")
include("markov_chain_hammer_imports.jl")
using CUDA
arraytype = CuArray
Ω = S¹(4π)^2
N = 2^7 # number of gridpoints
M = 2 # number of states
U = 1 # amplitdue factor
Q = ou_transition_matrix(M - 1)
A = collect(range(-sqrt(M - 1), sqrt(M - 1), length=M))
p = steady_state(Q)

grid = FourierGrid(N, Ω, arraytype=arraytype)
nodes, wavenumbers = grid.nodes, grid.wavenumbers



x = nodes[1]
y = nodes[2]
kˣ = wavenumbers[1]
kʸ = wavenumbers[2]

##
# Fields 
# velocity
ψ = arraytype(zeros(ComplexF64, N, N))
u = similar(ψ)
v = similar(ψ)

# theta
θs = [similar(ψ) for i in 1:M]
∂ˣθ = similar(ψ)
∂ʸθ = similar(ψ)
κΔθ = similar(ψ)
θ̇s = [similar(ψ) .* 0 for i in 1:M]
s = similar(ψ)
θ̅ = similar(ψ)
k₁ = similar(ψ)
k₂ = similar(ψ)
k₃ = similar(ψ)
k₄ = similar(ψ)
θ̃ = similar(ψ)
uθ = similar(ψ)
vθ = similar(ψ)
∂ˣuθ = similar(ψ)
∂ʸvθ = similar(ψ)

# source
s = similar(ψ)
@. s = cos(kˣ[2] * x) * cos(kʸ[2] * y) # could also set source term to zero

# operators
∂x = im * kˣ
∂y = im * kʸ
Δ = @. ∂x^2 + ∂y^2

# plan ffts
P = plan_fft!(ψ)
P⁻¹ = plan_ifft!(ψ)

##
Δx = x[2] - x[1]
κ = 0.01
cfl = 0.1
Δx = (x[2] - x[1])
advective_Δt = cfl * Δx / U
diffusive_Δt = cfl * Δx^2 / κ

# set stream function and hence velocity
@. ψ = cos(kˣ[2] * x) * cos(kʸ[2] * y)
P * ψ # in place fft
# ∇ᵖψ
@. u = -1.0 * (∂y * ψ); @. v = (∂x * ψ);
P⁻¹ * ψ; P⁻¹ * u; P⁻¹ * v; # don't need ψ anymore

simulation_parameters = (; A, p, Q, u, v, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ)

function n_state_rhs_symmetric!(θ̇s, θs, simulation_parameters)
    (; A, p, Q, u, v, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ) = simulation_parameters

    # need A (amplitude), p (probability of being in state), Q (transition probability)
    for (i, θ) in enumerate(θs)
        Aⁱ = A[i]; pⁱ = p[i]; θ̇ = θ̇s[i]
        # dynamics
        P * θ # in place fft
        # ∇θ
        @. ∂ˣθ = ∂x * θ; @. ∂ʸθ = ∂y * θ
        # κΔθ
        @. κΔθ = κ * Δ * θ
        # go back to real space 
        [P⁻¹ * field for field in (θ, ∂ˣθ, ∂ʸθ, κΔθ)] # in place ifft
        # compute u * θ and v * θ take derivative and come back
        @. uθ = u * θ; @. vθ = v * θ
        P * uθ; P * vθ
        @. ∂ˣuθ = ∂x * uθ; @. ∂ʸvθ = ∂y * vθ
        P⁻¹ * ∂ˣuθ; P⁻¹ * ∂ʸvθ
        # compute θ̇ in real space
        @. θ̇ = -Aⁱ * (u * ∂ˣθ + v * ∂ʸθ + ∂ˣuθ + ∂ʸvθ) * 0.5 + κΔθ + s * pⁱ
        # transitions
        for (j, θ2) in enumerate(θs)
            Qⁱʲ = Q[i, j]
            θ̇ .+= Qⁱʲ * θ2
        end
    end

    return nothing
end

n_state_rhs_symmetric!(θ̇s, θs, simulation_parameters)