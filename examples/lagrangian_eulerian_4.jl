using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2
# using GLMakie, HDF5
using ProgressBars
rng = MersenneTwister(1234)
Random.seed!(123456789)

include("transform.jl")
include("random_phase_kernel.jl")
# using GLMakie
save_fields = false
using CUDA
arraytype = CuArray
Ω = S¹(4π) × S¹(4π) # domain
N = (2^7, 2^7)      # number of gridpoints

# for (di, amplitude_factor) in ProgressBar(enumerate([0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]))
di = 1
amplitude_factor = 10.0
phase_speed = 1.0

grid = FourierGrid(N, Ω, arraytype=arraytype)
nodes, wavenumbers = grid.nodes, grid.wavenumbers

x = nodes[1]
y = nodes[2]
kˣ = wavenumbers[1]
kʸ = wavenumbers[2]
# construct filter
kxmax = maximum(kˣ)
kymax = maximum(kˣ)
filter = @. abs(kˣ) .+ 0 * abs(kʸ) ≤ 2 / 3 * kxmax
@. filter = filter * (0 * abs(kˣ) .+ 1 * abs(kʸ) ≤ 2 / 3 * kxmax)
filter = @. abs(kˣ) .+ 0 * abs(kʸ) ≤ Inf

# now define the random field 
wavemax = 3 # 3
𝓀 = arraytype(collect(-wavemax:0.5:wavemax))
𝓀ˣ = reshape(𝓀, (length(𝓀), 1))
𝓀ʸ = reshape(𝓀, (1, length(𝓀)))
A = @. (𝓀ˣ * 𝓀ˣ + 𝓀ʸ * 𝓀ʸ)^(-1.0) # @. (𝓀ˣ * 𝓀ˣ + 𝓀ʸ * 𝓀ʸ)^(-1)
A[A.==Inf] .= 0.0
φ = arraytype(2π * rand(size(A)...))
field = arraytype(zeros(N[1], N[2]))

##
# Fields 
# velocity
ψ = arraytype(zeros(ComplexF64, N[1], N[2]))
u = similar(ψ)
v = similar(ψ)

# theta
θ = similar(ψ)
∂ˣθ = similar(ψ)
∂ʸθ = similar(ψ)
κΔθ = similar(ψ)
θ̇ = similar(ψ)
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
φ .= 0.0
event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
wait(event)
P * ψ # in place fft
# ∇ᵖψ
@. u = -1.0 * (∂y * ψ)
@. v = (∂x * ψ)
# go back to real space 
P⁻¹ * ψ
P⁻¹ * θ
P⁻¹ * u
P⁻¹ * v
u₀ = sqrt(real(mean(u .* u)))
v₀ = sqrt(real(mean(v .* v)))
A .*= amplitude_factor * sqrt(2) / u₀
# check it 
event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
wait(event)
P * ψ # in place fft
# ∇ᵖψ
@. u = -1.0 * (∂y * ψ)
@. v = (∂x * ψ)
# go back to real space 
P⁻¹ * ψ
P⁻¹ * θ
P⁻¹ * u
P⁻¹ * v
u₀ = sqrt(real(mean(u .* u))) # / sqrt(2)
v₀ = sqrt(real(mean(v .* v))) # / sqrt(2)



function rhs_lagrangian!(du, u, parameters)
    #=
    du[1] = 1.0 * cos(u[1]) * sin(u[2]) * A
    du[2] = -1.0 * sin(u[1]) * cos(u[2]) * A
    =#
    (; A, 𝓀ˣ, 𝓀ʸ, φ) = parameters
    du[1] = 0.0
    du[2] = 0.0
    xx = u[1]
    yy = u[2]
    for i in eachindex(𝓀ˣ), j in eachindex(𝓀ʸ)
        # tmp_sum += A[i, j] * cos(𝓀ˣ[i] * xx + 𝓀ʸ[j] * yy + φ[i, j]) # stream function 
        tmp = A[i, j] * sin(𝓀ˣ[i] * xx + 𝓀ʸ[j] * yy + φ[i, j])
        du[1] +=  𝓀ʸ[j] * tmp # stream function
        du[2] -=  𝓀ˣ[i] * tmp # stream function
    end

    return nothing
end

function rk4(f, s, dt, parameters)
    ls = length(s)
    k1 = zeros(ls)
    k2 = zeros(ls)
    k3 = zeros(ls)
    k4 = zeros(ls)
    f(k1, s, parameters)
    f(k2, s + k1 * dt / 2, parameters)
    f(k3, s + k2 * dt / 2, parameters)
    f(k4, s + k3 * dt, parameters)
    return s + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
end


Δt = 0.01 / amplitude_factor
iend = floor(Int, 100 * 5 * amplitude_factor)

u²_lagrangian_particle = zeros(iend)
uv_lagrangian_particle = zeros(iend)
vu_lagrangian_particle = zeros(iend)
v²_lagrangian_particle = zeros(iend)

s = zeros(2) 
ṡ = zeros(2)
A = Array(A)
𝓀ˣ = Array(𝓀ˣ)
𝓀ʸ = Array(𝓀ʸ)
φ = Array(φ)
parameters = (; A, 𝓀ˣ, 𝓀ʸ, φ)
realizations = 10000

for j in ProgressBar(1:realizations)
    i_index = 64
    j_index = 64

    s[1] = 2π
    s[2] = 2π

    rand!(rng, φ)
    φ .*= 2π

    rhs_lagrangian!(ṡ, s, parameters)
    u0 = copy(ṡ[1])
    v0 = copy(ṡ[2])

    for i = 1:iend
        # need to account for the flow field changing in time
        uτL = copy(ṡ[1])
        vτL = copy(ṡ[2])
        rhs_lagrangian!(ṡ, s, parameters)

        # correlate current with past
        u²_lagrangian_particle[i] += real.(uτL .* u0) ./ realizations
        uv_lagrangian_particle[i] += real.(uτL .* v0) ./ realizations
        vu_lagrangian_particle[i] += real.(vτL .* u0) ./ realizations
        v²_lagrangian_particle[i] += real.(vτL .* v0) ./ realizations

        # fourth order runge kutta on deterministic part
        snew = rk4(rhs_lagrangian!, s, Δt, parameters)
        s .= (snew .% 4π) # just in case

        φ̇ = randn(size(φ)) # gaussian noise
        @. φ += sqrt(2 * Δt) * φ̇

    end
end


i_index = 64
j_index = 64
fig3 = Figure()
ax1 = fig3[1, 1] = Axis(fig3, xlabel="time", ylabel="u²")
ylims!(ax1, -0.2 * amplitude_factor^2, 1.2 * amplitude_factor^2)
tlist = collect(0:iend-1) .* Δt
eulerian_decorrelation = @. amplitude_factor^2 * exp(-tlist)
lines!(ax1, tlist, u²_lagrangian_particle[:], color=:green, linewidth=3)
lines!(ax1, tlist, eulerian_decorrelation, color=:red, linewidth=3)
ax2 = fig3[1, 2] = Axis(fig3, xlabel="time", ylabel="uv")
ylims!(ax2, minimum(vu_lagrangian_particle[:]) * 1.1, maximum(uv_lagrangian_particle[:]) * 1.1)
lines!(ax2, uv_lagrangian_particle[:], color=:green, linewidth=3)
ax3 = fig3[2, 1] = Axis(fig3, xlabel="time", ylabel="vu")
ylims!(ax3, minimum(vu_lagrangian_particle[:]) * 1.1, maximum(vu_lagrangian_particle[:]) * 1.1)
lines!(ax3, vu_lagrangian_particle[:], color=:green, linewidth=3)
ax4 = fig3[2, 2] = Axis(fig3, xlabel="time", ylabel="v²")
ylims!(ax4, -0.2 * amplitude_factor^2, 1.2 * amplitude_factor^2)
lines!(ax4, tlist, v²_lagrangian_particle[:], color=:green, linewidth=3)
lines!(ax4, tlist, eulerian_decorrelation, color=:red, linewidth=3)