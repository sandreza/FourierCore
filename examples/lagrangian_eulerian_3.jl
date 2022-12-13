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
Nens = 2 # here this means that we initialize with u or initialize with v
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
@. s = sin(kˣ[index] * x) * sin(kʸ[index] * y) * 0# could also set source term to zero

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

rhs! = n_state_rhs_ensemble_symmetric!

println("starting simulations")

tend = 5.0
iend = ceil(Int, tend / Δt)

realizations = 1000

u²_lagrangian = zeros(N, N, iend)
uv_lagrangian = copy(u²_lagrangian)
vu_lagrangian = copy(u²_lagrangian)
v²_lagrangian = copy(u²_lagrangian)
u²_eulerian = copy(u²_lagrangian)
uv_eulerian = copy(u²_lagrangian) # only need uv in this case
vu_eulerian = copy(u²_lagrangian) # only need uv in this case
v²_eulerian = copy(u²_lagrangian)

PF = exp(Q * Δt)
tmpAs = []
for j in ProgressBar(1:realizations)
    tmpA = generate(PF, iend)
    push!(tmpAs, tmpA)
    A_indices = [tmpA for i in 1:Nens]

    [A[k] = Us[A_indices[k][1]] for k in eachindex(A_indices)]
    # initialize with velocity field
    θs[1] .= A[1] * u
    θs[2] .= A[2] * v

    # remember the initial condition
    u0 = copy(θs[1])
    v0 = copy(θs[2])

    for i = 1:iend
        uτL = copy(θs[1])
        vτL = copy(θs[2])

        # correlate current with past
        u²_lagrangian[:, :, i] .+= real.(Array(uτL .* u0)) ./ realizations
        uv_lagrangian[:, :, i] .+= real.(Array(uτL .* v0)) ./ realizations
        vu_lagrangian[:, :, i] .+= real.(Array(vτL .* u0)) ./ realizations
        v²_lagrangian[:, :, i] .+= real.(Array(vτL .* v0)) ./ realizations

        u²_eulerian[:, :, i] .+= A[1] * real.(Array(u .* u0)) ./ realizations
        uv_eulerian[:, :, i] .+= A[1] * real.(Array(u .* v0)) ./ realizations
        vu_eulerian[:, :, i] .+= A[1] * real.(Array(v .* u0)) ./ realizations
        v²_eulerian[:, :, i] .+= A[1] * real.(Array(v .* v0)) ./ realizations

        [A[k] = Us[A_indices[k][i]] for k in eachindex(A_indices)]
        # fourth order runge kutta on deterministic part
        # keep ψ frozen is the correct way to do it here
        rhs!(k₁, θs, simulation_parameters)
        [θ̃[i] .= θs[i] .+ Δt * k₁[i] * 0.5 for i in eachindex(θs)]
        rhs!(k₂, θ̃, simulation_parameters)
        [θ̃[i] .= θs[i] .+ Δt * k₂[i] * 0.5 for i in eachindex(θs)]
        rhs!(k₃, θ̃, simulation_parameters)
        [θ̃[i] .= θs[i] .+ Δt * k₃[i] * 0.5 for i in eachindex(θs)]
        rhs!(k₄, θ̃, simulation_parameters)
        [θs[i] .+= Δt / 6 * (k₁[i] + 2 * k₂[i] + 2 * k₃[i] + k₄[i]) for i in eachindex(θs)]
    end
end

#=
using HDF5
filename = "lagrangian_vs_eulerian" * "_amp_" * string(amplitude_factor) * "_members_" * string(realizations) * ".hdf5"
fid = h5open(filename, "w")
fid["molecular_diffusivity"] = κ
fid["streamfunction_amplitude"] = Array(A)
fid["phase increase"] = phase_speed
fid["time"] = collect(Δt * (2:iend))
fid["lagrangian"] = lagrangian_list[1:end-1]
fid["eulerian"] = eulerian_list[2:end]
fid["ensemble number"] = realizations
close(fid)
=#
##

## 
# exact eulerian diffusivity 
fig = Figure()
ax11 = Axis(fig[1, 1]; xlabel="x", ylabel="y", title="exact eulerian u²", xlabelsize=30, ylabelsize=30)
ax12 = Axis(fig[1, 2]; xlabel="x", ylabel="y", title="exact eulerian uv", xlabelsize=30, ylabelsize=30)
ax13 = Axis(fig[1, 3]; xlabel="x", ylabel="y", title="exact eulerian vu", xlabelsize=30, ylabelsize=30)
ax14 = Axis(fig[1, 4]; xlabel="x", ylabel="y", title="exact eulerian v²", xlabelsize=30, ylabelsize=30)
u² = Array(real.(u .* u))
uv = Array(real.(u .* v))
vu = Array(real.(v .* u))
v² = Array(real.(v .* v))
contourf!(ax11, xA, yA, u², colormap=:plasma)
contourf!(ax12, xA, yA, uv, colormap=:balance)
contourf!(ax13, xA, yA, vu, colormap=:balance)
contourf!(ax14, xA, yA, v², colormap=:plasma)

#below was autocompleted
ax21 = Axis(fig[2, 1]; xlabel="x", ylabel="y", title="lagrangian u²", xlabelsize=30, ylabelsize=30)
ax22 = Axis(fig[2, 2]; xlabel="x", ylabel="y", title="lagrangian uv", xlabelsize=30, ylabelsize=30)
ax23 = Axis(fig[2, 3]; xlabel="x", ylabel="y", title="lagrangian vu", xlabelsize=30, ylabelsize=30)
ax24 = Axis(fig[2, 4]; xlabel="x", ylabel="y", title="lagrangian v²", xlabelsize=30, ylabelsize=30)
contourf!(ax21, xA, yA, Δt * sum(u²_lagrangian, dims=3)[:, :, 1], colormap=:plasma)
contourf!(ax22, xA, yA, Δt * sum(uv_lagrangian, dims=3)[:, :, 1], colormap=:balance)
contourf!(ax23, xA, yA, Δt * sum(vu_lagrangian, dims=3)[:, :, 1], colormap=:balance)
contourf!(ax24, xA, yA, Δt * sum(v²_lagrangian, dims=3)[:, :, 1], colormap=:plasma)

ax31 = Axis(fig[3, 1]; xlabel="x", ylabel="y", title="eulerian u²", xlabelsize=30, ylabelsize=30)
ax32 = Axis(fig[3, 2]; xlabel="x", ylabel="y", title="eulerian uv", xlabelsize=30, ylabelsize=30)
ax33 = Axis(fig[3, 3]; xlabel="x", ylabel="y", title="eulerian vu", xlabelsize=30, ylabelsize=30)
ax34 = Axis(fig[3, 4]; xlabel="x", ylabel="y", title="eulerian v²", xlabelsize=30, ylabelsize=30)
contourf!(ax31, xA, yA, Δt * sum(u²_eulerian, dims=3)[:, :, 1], colormap=:plasma)
contourf!(ax32, xA, yA, Δt * sum(uv_eulerian, dims=3)[:, :, 1], colormap=:balance)
contourf!(ax33, xA, yA, Δt * sum(vu_eulerian, dims=3)[:, :, 1], colormap=:balance)
contourf!(ax34, xA, yA, Δt * sum(v²_eulerian, dims=3)[:, :, 1], colormap=:plasma)

##
fig2 = Figure()
for i in 1:4
    for j in 1:4
        ax = Axis(fig2[i, j]; title = "index ($i,$j)")
        index1 = 1 + (i-1) * 10
        index2 = 96 - (j-1) * 10
        tmp = u²_eulerian[index1, index2, :]
        lines!(ax, tmp, color=:red, linewidth = 3)
        lines!(ax, tmp[1] * exp.(-collect(0:iend-1) .* Δt), color=:orange, linewidth = 3)
        lines!(ax, u²_lagrangian[index1, index2, :], color=:blue, linewidth =3 )
    end
end

## 
# Lagrangian partical calculation 
# ψ = cos(kˣ[2] * x) * cos(kʸ[2] * y)
# u = -1.0 * (∂y * ψ); => u = kʸ[2] * cos(kˣ[2] * x) * sin(kʸ[2] * y)
# v = (∂x * ψ); => v = -1.0 * kˣ[2] * sin(kˣ[2] * x) * cos(kʸ[2] * y)
# kʸ[2] = kˣ[2] = 1.0
function rhs_lagrangian!(du, u, A)
    du[1] =  1.0 * cos(u[1]) * sin(u[2]) * A
    du[2] = -1.0 * sin(u[1]) * cos(u[2]) * A
    return nothing
end
function rk4(f, s, dt, A)
    ls = length(s)
    k1 = zeros(ls)
    k2 = zeros(ls)
    k3 = zeros(ls)
    k4 = zeros(ls)
    f(k1, s, A)
    f(k2, s + k1 * dt / 2, A)
    f(k3, s + k2 * dt / 2, A)
    f(k4, s + k3 * dt, A)
    return s + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
end

u_A = Array(u)
v_A = Array(v)
x_A = Array(x)
y_A = Array(y)

u²_lagrangian_particle = zeros(iend)
uv_lagrangian_particle = zeros(iend)
vu_lagrangian_particle = zeros(iend)
v²_lagrangian_particle = zeros(iend)

s = zeros(2)
tmpA2s = []

for j in ProgressBar(1:realizations)
    tmpA = generate(PF, iend)
    push!(tmpA2s, tmpA)
    U_history = Us[tmpA]
    # initialize with velocity field
    i_index = 1
    j_index = 96
    s[1] = x_A[i_index]
    s[2] = y_A[j_index]
    # remember the initial condition
    u0 = real(u_A[i_index, j_index])
    v0 = real(v_A[i_index, j_index])
    for i = 1:iend
        uτL = copy(s[1])
        vτL = copy(s[2])
        i_index_now = argmin(abs.(s[1] .- x_A[:]))
        j_index_now = argmin(abs.(s[2] .- y_A[:]))

        uτL = real(u_A[i_index_now, j_index_now])
        vτL = real(v_A[i_index_now, j_index_now])

        # correlate current with past
        u²_lagrangian_particle[i] += real.(uτL .* u0) ./ realizations
        uv_lagrangian_particle[i] += real.(uτL .* v0) ./ realizations
        vu_lagrangian_particle[i] += real.(vτL .* u0) ./ realizations
        v²_lagrangian_particle[i] += real.(vτL .* v0) ./ realizations

        # fourth order runge kutta on deterministic part
        snew = rk4(rhs_lagrangian!, s, Δt, U_history[i])
        s .= snew

    end
end


##
#=
tmpUs = []
for j in ProgressBar(1:realizations)
    tmpU = Us[generate(PF, iend)]
    push!(tmpUs, tmpU)
end
first_state = [tmpUs[1][i] for i in 1:iend]
acor = zeros(iend)
for j in ProgressBar(1:realizations)
    for i in 1:iend
        acor[i] += tmpUs[j][i] * tmpUs[j][1] / realizations
    end
end
fig3 = Figure()
ax = Axis(fig3[1,1])
scatter!(ax, acor, color = :red)
scatter!(ax, exp.( -collect(0:iend-1) .* Δt), color = :blue)
=#