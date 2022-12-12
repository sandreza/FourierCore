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
Ω = S¹(2π)^2
N = 2^7 # number of gridpoints
M = 2 # number of states
Nens = 1000 # number of ensemble members
U = 1.0 # amplitude factor
Q = ou_transition_matrix(M - 1)
Us = U .* collect(range(-sqrt(M - 1), sqrt(M - 1), length=M))
A = zeros(Nens)
p = steady_state(Q)
Λ, V = eigen(Q)
V[:, end] .= p
V⁻¹ = inv(V)

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
θs = [similar(ψ) for i in 1:Nens]
∂ˣθ = similar(ψ)
∂ʸθ = similar(ψ)
κΔθ = similar(ψ)
θ̇s = [similar(ψ) .* 0 for i in 1:Nens]
s = similar(ψ)
θ̅ = similar(ψ)
k₁ = [similar(ψ) for i in 1:Nens]
k₂ = [similar(ψ) for i in 1:Nens]
k₃ = [similar(ψ) for i in 1:Nens]
k₄ = [similar(ψ) for i in 1:Nens]
θ̃ = [similar(ψ) for i in 1:Nens]
uθ = similar(ψ)
vθ = similar(ψ)
∂ˣuθ = similar(ψ)
∂ʸvθ = similar(ψ)

# source
s = similar(ψ)
index = 2
@. s = sin(kˣ[index] * x) * sin(kʸ[index] * y) # could also set source term to zero

# operators
∂x = im * kˣ
∂y = im * kʸ
Δ = @. ∂x^2 + ∂y^2
κ = 0.01

# set equal to diffusive solution 
tmp = (kˣ[index]^2 + kʸ[index]^2)
for (i, θ) in enumerate(θs)
    θ .= (s ./ (tmp * κ))
end

println("maximum value of theta before ", maximum(real.(sum(θs))))


# plan ffts
P = plan_fft!(ψ)
P⁻¹ = plan_ifft!(ψ)

# set stream function and hence velocity
@. ψ = cos(kˣ[2] * x) * cos(kʸ[2] * y)
P * ψ # in place fft
# ∇ᵖψ
@. u = -1.0 * (∂y * ψ);
@. v = (∂x * ψ);
P⁻¹ * ψ;
P⁻¹ * u;
P⁻¹ * v; # don't need ψ anymore

## timestepping
Δx = x[2] - x[1]
cfl = 0.1 #0.1
advective_Δt = cfl * Δx / maximum(real.(u))
diffusive_Δt = cfl * Δx^2 / κ
Δt = min(advective_Δt, diffusive_Δt)

simulation_parameters = (; A, u, v, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ)

function n_state_rhs_ensemble_symmetric!(θ̇s, θs, simulation_parameters)
    (; A, u, v, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ) = simulation_parameters

    # need A (amplitude), p (probability of being in state), Q (transition probability)
    for (i, θ) in enumerate(θs)
        Aⁱ = A[i]
        θ̇ = θ̇s[i]
        # dynamics
        P * θ # in place fft
        # ∇θ
        @. ∂ˣθ = ∂x * θ
        @. ∂ʸθ = ∂y * θ
        # κΔθ
        @. κΔθ = κ * Δ * θ
        # go back to real space 
        [P⁻¹ * field for field in (θ, ∂ˣθ, ∂ʸθ, κΔθ)] # in place ifft
        # compute u * θ and v * θ take derivative and come back
        @. uθ = u * θ
        @. vθ = v * θ
        P * uθ
        P * vθ
        @. ∂ˣuθ = ∂x * uθ
        @. ∂ʸvθ = ∂y * vθ
        P⁻¹ * ∂ˣuθ
        P⁻¹ * ∂ʸvθ
        # compute θ̇ in real space
        @. θ̇ = -Aⁱ * (u * ∂ˣθ + v * ∂ʸθ + ∂ˣuθ + ∂ʸvθ) * 0.5 + κΔθ + s
    end

    return nothing
end

n_state_rhs_ensemble_symmetric!(θ̇s, θs, simulation_parameters)

rhs! = n_state_rhs_ensemble_symmetric!

tend = 100.0
iend = ceil(Int, tend / Δt)

PF = exp(Q * Δt)
A_indices = [generate(PF, iend) for i in 1:Nens]
# runge kutta 4 timestepping
for i in ProgressBar(1:iend)
    [A[k] = Us[A_indices[k][i]] for k in eachindex(A_indices)]
    rhs!(k₁, θs, simulation_parameters)
    [θ̃[i] .= θs[i] .+ Δt * k₁[i] * 0.5 for i in eachindex(θs)]
    rhs!(k₂, θ̃, simulation_parameters)
    [θ̃[i] .= θs[i] .+ Δt * k₂[i] * 0.5 for i in eachindex(θs)]
    rhs!(k₃, θ̃, simulation_parameters)
    [θ̃[i] .= θs[i] .+ Δt * k₃[i] * 0.5 for i in eachindex(θs)]
    rhs!(k₄, θ̃, simulation_parameters)
    [θs[i] .+= Δt / 6 * (k₁[i] + 2 * k₂[i] + 2 * k₃[i] + k₄[i]) for i in eachindex(θs)]
end

println("maximum value of ensemble mean theta after ", maximum(real.(mean(θs))))

# plot

using GLMakie
xA = Array(x)[:]
yA = Array(y)[:]
fig = Figure()
colorrange = (-1, 1)
for i in eachindex(θs)
    if i < 6
        ax = Axis(fig[1, i]; title="state $i")
        θA = real.(Array(θs[i]))
        contourf!(ax, xA, yA, θA, colormap=:balance)
    end

end

θA = Array(real.(mean(θs)))
ax2 = Axis(fig[2, 1]; title="mean state")
contourf!(ax2, xA, yA, θA, colormap=:balance)

display(fig)

last_time = [A_indices[k][end] for k in eachindex(A_indices)]
index1 = last_time .== 1
index2 = last_time .== 2
θA1 = Array(real.(mean(θs[index1]))) * p[1]
θA2 = Array(real.(mean(θs[index2]))) * p[2]

fig = Figure()
ax1 = Axis(fig[1, 1]; title="weighted average 1")
ax2 = Axis(fig[1, 2]; title="weighted average 2")
contourf!(ax1, xA, yA, θA1, colormap=:balance)
contourf!(ax2, xA, yA, θA2, colormap=:balance)

using GLMakie
xA = Array(x)[:]
yA = Array(y)[:]
fig = Figure()
PWCAs = [] #probability weighted conditional average
Spectral_States = [] #probability weighted conditional average
last_time = [A_indices[k][end] for k in eachindex(A_indices)]
for i in 1:M
    ax = Axis(fig[1, i]; title="pwca $i")
    index = (last_time .== i)
    θA = Array(real.(mean(θs[index]))) * p[i]
    push!(PWCAs, copy(θA))

    contourf!(ax, xA, yA, θA, colormap=:balance)
end
for i in 1:M
    θA .= 0.0
    ax2 = Axis(fig[2, i]; title="spectral state $i")
    for j in 1:M
        θA .+= real.(Array(PWCAs[j])) .* V⁻¹[end-i+1, j]
    end
    push!(Spectral_States, copy(θA))
    contourf!(ax2, xA, yA, θA, colormap=:balance)
end
display(fig)


using HDF5
fid = h5open("states_" * string(M) * "_ensemble_members_" * string(Nens) * ".hdf5", "w")
fid["molecular_diffusivity"] = κ
fid["time"] = collect(Δt * iend)
fid["velocity amplitudes"] = Us 
fid["stream function"] = Array(real.(ψ))
fid["generator"] = Q
fid["source"] = real.(Array(s))
for i in eachindex(θs)
    fid["$i"] = real.(Array(θs[i]))
    fid["history $i"] = A_indices[i]
end
for i in 1:M
    fid["pwca $i"] = PWCAs[i]
    fid["spectral state $i"] = Spectral_States[i]
end
close(fid)

#=
last_time = [A_indices[k][end]] for k in eachindex(A_indices)]
=#