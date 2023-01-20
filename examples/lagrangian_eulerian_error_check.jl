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
amplitude_factor = 1.0
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

# now define the random field 
wavemax = 3 # 3
𝓀 = arraytype(collect(-wavemax:0.5:wavemax))
𝓀ˣ = reshape(𝓀, (length(𝓀), 1))
𝓀ʸ = reshape(𝓀, (1, length(𝓀)))
A = @. (𝓀ˣ * 𝓀ˣ + 𝓀ʸ * 𝓀ʸ)^(-1.0) # @. (𝓀ˣ * 𝓀ˣ + 𝓀ʸ * 𝓀ʸ)^(-1)
# A = @. exp(-(𝓀ˣ * 𝓀ˣ + 𝓀ʸ * 𝓀ʸ))
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
        du[1] += 𝓀ʸ[j] * tmp # stream function
        du[2] -= 𝓀ˣ[i] * tmp # stream function
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
timeend = 1
iend = floor(Int, 100 * timeend * amplitude_factor)

# eulerian calculation 
phases = []
rand!(rng, φ)
φ .*= 2π
push!(phases, Array(φ))
for i in 1:iend
    φ̇ = CuArray(randn(size(φ))) # gaussian noise
    @. φ += sqrt(2 * Δt) * φ̇
    push!(phases, Array(φ))
end

s .*= false
κ = 0.01
simulation_parameters = (; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, filter=nothing, ∂x, ∂y, κ, Δ, κΔθ)
φ .= CuArray(phases[1])
event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
wait(event)
# get horizontal velocity
P * ψ
u0 = -∂y .* ψ
P⁻¹ * ψ
P⁻¹ * u0

θ .= u0

rhs! = θ_rhs_symmetric_zeroth!

for i = 1:iend
    # fourth order runge kutta on deterministic part
    # keep ψ frozen is the correct way to do it here

    # the below assumes that φ is just a function of time
    rhs!(k₁, θ, simulation_parameters)
    @. θ̃ = θ + Δt * k₁ * 0.5
    rhs!(k₂, θ̃, simulation_parameters)
    @. θ̃ = θ + Δt * k₂ * 0.5
    rhs!(k₃, θ̃, simulation_parameters)
    @. θ̃ = θ + Δt * k₃
    rhs!(k₄, θ̃, simulation_parameters)
    @. θ += real(Δt / 6 * (k₁ + 2 * k₂ + 2 * k₃ + k₄))
    φ .= CuArray(phases[i+1])

end
uend = real.(Array(θ))

## lagrangian particle 

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
realizations = 1

xskip = 1
x_particles = Array(x)[1:xskip:end]
y_particles = Array(y)[1:xskip:end]

x_array = zeros(length(x_particles), length(y_particles), iend + 1)
y_array = zeros(length(x_particles), length(y_particles), iend + 1)

u_array = zeros(length(x_particles), length(y_particles), iend + 1)
v_array = zeros(length(x_particles), length(y_particles), iend + 1)

for (ii, xval) in ProgressBar(enumerate(x_particles))
    for (jj, yval) in enumerate(y_particles)
        for j in 1:realizations
            s[1] = xval
            s[2] = yval

            φ .= phases[1]
            rhs_lagrangian!(ṡ, s, parameters)
            u0 = copy(ṡ[1])
            v0 = copy(ṡ[2])

            u_array[ii, jj, 1] = u0
            v_array[ii, jj, 1] = v0

            x_array[ii, jj, 1] = xval
            y_array[ii, jj, 1] = yval

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

                φ .= phases[i+1]
                x_array[ii, jj, i+1] = s[1]
                y_array[ii, jj, i+1] = s[2]

                uτL = copy(ṡ[1])
                vτL = copy(ṡ[2])

                u_array[ii, jj, i+1] = uτL
                v_array[ii, jj, i+1] = vτL

            end
        end
    end
end

#=
using GLMakie
tskip = 10
fig = Figure()
ax11 = Axis(fig[1, 1]; title="t = 0")
ax12 = Axis(fig[1, 2]; title="t = 1/3")
ax21 = Axis(fig[2, 1]; title="t = 2/3")
ax22 = Axis(fig[2, 2]; title="t = 1")

scatter!(ax11, x_array[:, :, 1][:], y_array[:, :, 1][:], color=:red)
scatter!(ax12, x_array[:, :, 1:tskip:33][:], y_array[:, :, 1:tskip:33][:], color=:red)
scatter!(ax21, x_array[:, :, 1:tskip:66][:], y_array[:, :, 1:tskip:66][:], color=:red)
scatter!(ax22, x_array[:, :, 1:tskip:100][:], y_array[:, :, 1:tskip:100][:], color=:red)
=#

tmp = [argmin(abs.(x_array[:, :, end] .- x_array[i, j, 1]) .+ abs.(y_array[:, :, end] .- y_array[i, j, 1])) for i in 1:16, j in 1:16]


x_array[tmp[10, 10][1], tmp[10, 10][2], end]
x_array[10, 10, 1]

x_array[tmp[10, 10][1], tmp[10, 10][2], 1]
y_array[tmp[10, 10][1], tmp[10, 10][2], 1]

tmpu0 = Array(real.(u0))[1:xskip:end, 1:xskip:end]

tmput = uend[1:xskip:end, 1:xskip:end]

# The value of (X(t), Y(t)) ends up at a given position. The index 10,10 is the grid position
tmpu0[tmp[10, 10][1]-1:tmp[10, 10][1]+1, tmp[10, 10][2]-1:tmp[10, 10][2]+1, 1]
tmput[10-1:10+1, 10-1:10+1]