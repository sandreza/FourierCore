using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2, HDF5
using ProgressBars
using CUDA
# using GLMakie

rng = MersenneTwister(1234)
Random.seed!(123456789)

include("transform.jl")
include("random_phase_kernel.jl")

save_fields = false

arraytype = CuArray
Ω = S¹(4π)^2
N = 2^7 # number of gridpoints
phase_speed = 1.0
amplitude_factor = 0.5

#=
filename = "effective_diffusivities_samples_100.h5"
fid = h5open(filename, "w")
create_group(fid, "effective_diffusivities")
create_group(fid, "amplitude_factor")
=#

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
wavemax = 3
𝓀 = arraytype(collect(-wavemax:0.5:wavemax))
𝓀ˣ = reshape(𝓀, (length(𝓀), 1))
𝓀ʸ = reshape(𝓀, (1, length(𝓀)))
A = @. (𝓀ˣ * 𝓀ˣ + 𝓀ʸ * 𝓀ʸ)^(-1)
A[A.==Inf] .= 0.0
φ = arraytype(2π * rand(size(A)...))
field = arraytype(zeros(N, N))

##
# Fields 
# velocity
ψ = arraytype(zeros(ComplexF64, N, N))
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
s¹ = similar(ψ)

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

##
# κ = 2 / N  # roughly 1/N for this flow
# κ = 2 / 2^8 # fixed diffusivity
# κ = 2e-4
Δx = x[2] - x[1]
κ = 0.01 * (2^7 / N)^2# amplitude_factor * 2 * Δx^2
cfl = 0.1
Δx = (x[2] - x[1])
advective_Δt = cfl * Δx / amplitude_factor
diffusive_Δt = cfl * Δx^2 / κ
Δt = minimum([advective_Δt, diffusive_Δt])

# take the initial condition as negative of the source
maxind = minimum([40, floor(Int, N / 4)])
index_choices = 2:2
tic = Base.time()

@. s *= false
@. s¹ *= false

for index_choice in ProgressBar(index_choices)
    kᶠ = kˣ[index_choice]
    @. s¹ += cos(kᶠ * x)
end

t = [0.0]
tend = 100 # 5000

iend = ceil(Int, tend / Δt)


φ .= arraytype(2π * rand(size(A)...))
θ .*= 0.0
θ̅ .*= 0.0
size_of_A = size(A)

rhs! = θ_rhs_symmetric_ensemble!

ensemble_size = 400
ψs = [arraytype(zeros(ComplexF64, N, N)) for i in 1:ensemble_size]
θs = [arraytype(zeros(ComplexF64, N, N)) for i in 1:ensemble_size]
φs = [arraytype(2π * rand(size_of_A...)) for i in 1:ensemble_size]
ensemble_indices = 1:ensemble_size

for ω in ProgressBar(ensemble_indices)
    θω = θs[ω]
    ψω = ψs[ω]
    φω = φs[ω]

    stream_function!(ψω, A, 𝓀ˣ, 𝓀ʸ, x, y, φω)
end

function ensemble_flux_divergence!(s, ψs, θs, u, v, uθ, vθ, ∂x, ∂y, ∂ˣuθ, ∂ʸvθ, P, P⁻¹, ensemble_size)
    ensemble_indices = 1:ensemble_size
    @. s *= false
    for ω in ensemble_indices
        ψω = ψs[ω]
        θω = θs[ω]
        P * ψω
        @. u = -1.0 * (∂y * ψω)
        @. v = (∂x * ψω)
        P⁻¹ * u
        P⁻¹ * v
        P⁻¹ * ψω
        @. uθ = u * θω
        @. vθ = v * θω
        P * uθ
        P * vθ
        @. ∂ˣuθ = ∂x * uθ
        @. ∂ʸvθ = ∂y * vθ
        P⁻¹ * ∂ˣuθ
        P⁻¹ * ∂ʸvθ
        @. s += real(∂ˣuθ + ∂ʸvθ) / ensemble_size
    end
    # take off mean component 
    ψω  = mean(ψs)
    θω = mean(θs)
    P * ψω
    @. u = -1.0 * (∂y * ψω)
    @. v = (∂x * ψω)
    P⁻¹ * u
    P⁻¹ * v
    P⁻¹ * ψω
    @. uθ = u * θω
    @. vθ = v * θω
    P * uθ
    P * vθ
    @. ∂ˣuθ = ∂x * uθ
    @. ∂ʸvθ = ∂y * vθ
    P⁻¹ * ∂ˣuθ
    P⁻¹ * ∂ʸvθ
    @. s -= real(∂ˣuθ + ∂ʸvθ)
    return nothing
end

function ensemble_flux_u!(s, ψs, θs, u, v, uθ, vθ, ∂x, ∂y, ∂ˣuθ, ∂ʸvθ, P, P⁻¹, ensemble_size)
    ensemble_indices = 1:ensemble_size
    @. s *= false
    for ω in ensemble_indices
        ψω = ψs[ω]
        θω = θs[ω]
        P * ψω
        @. u = -1.0 * (∂y * ψω)
        @. v = (∂x * ψω)
        P⁻¹ * u
        P⁻¹ * v
        P⁻¹ * ψω
        @. uθ = u * θω
        @. s += real(uθ) / ensemble_size
    end
    return nothing
end

function ensemble_flux_v!(s, ψs, θs, u, v, uθ, vθ, ∂x, ∂y, ∂ˣuθ, ∂ʸvθ, P, P⁻¹, ensemble_size)
    ensemble_indices = 1:ensemble_size
    @. s *= false
    for ω in ensemble_indices
        ψω = ψs[ω]
        θω = θs[ω]
        P * ψω
        @. u = -1.0 * (∂y * ψω)
        @. v = (∂x * ψω)
        P⁻¹ * u
        P⁻¹ * v
        P⁻¹ * ψω
        @. vθ = v * θω
        @. s += real(vθ) / ensemble_size
    end
    return nothing
end

# @. s *= false # incorrect
ensemble_flux_divergence!(s, ψs, θs, u, v, uθ, vθ, ∂x, ∂y, ∂ˣuθ, ∂ʸvθ, P, P⁻¹, ensemble_size)
for i = ProgressBar(1:iend)
    for ω in ensemble_indices
        θω = θs[ω]
        ψω = ψs[ω]
        φω = φs[ω]

        simulation_parameters = (; ψ=ψω, A, 𝓀ˣ, 𝓀ʸ, x, y, φ=φω, u, v, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, s¹, P, P⁻¹, filter)
        # the below assumes that φ is just a function of time
        rhs!(k₁, θω, simulation_parameters)
        @. θ̃ = θω

        φ_rhs_normal!(φ̇, φω, rng)

        @. φω += phase_speed * sqrt(Δt / 2) * φ̇

        rhs!(k₂, θ̃, simulation_parameters)
        @. θ̃ = θω + Δt * k₂ * 0.5
        rhs!(k₃, θ̃, simulation_parameters)
        @. θ̃ = θω + Δt * k₃

        φ_rhs_normal!(φ̇, φω, rng)
        @. φω += phase_speed * sqrt(Δt / 2) * φ̇

        rhs!(k₄, θ̃, simulation_parameters)
        @. θω += Δt / 6 * (k₁ + 2 * k₂ + 2 * k₃ + k₄)

        t[1] += Δt
    end

    ensemble_flux_divergence!(s, ψs, θs, u, v, uθ, vθ, ∂x, ∂y, ∂ˣuθ, ∂ʸvθ, P, P⁻¹, ensemble_size)

end


tmp = real.(Array(mean(θs)))
tmpsi = mean(ψs)
P * tmpsi
@. u = -1.0 * (∂y * tmpsi); @. v = (∂x * tmpsi)
P⁻¹ * u; P⁻¹ * v; P⁻¹ * tmpsi
tmpu = real.(Array(u))
tmpv = real.(Array(v))
tmpsi = real.(Array(tmpsi))

# heatmap(tmp)
ensemble_flux_divergence!(s, ψs, θs, u, v, uθ, vθ, ∂x, ∂y, ∂ˣuθ, ∂ʸvθ, P, P⁻¹, ensemble_size)
tmps = real.(Array(s))
# heatmap(tmps)

ensemble_flux_u!(s, ψs, θs, u, v, uθ, vθ, ∂x, ∂y, ∂ˣuθ, ∂ʸvθ, P, P⁻¹, ensemble_size)
tmpuc = real.(Array(s))
# heatmap(tmpu)

ensemble_flux_v!(s, ψs, θs, u, v, uθ, vθ, ∂x, ∂y, ∂ˣuθ, ∂ʸvθ, P, P⁻¹, ensemble_size)
tmpvc = real.(Array(s))
# heatmap(tmpv)

uc = Array(abs.(fft(mean(tmpuc, dims=2)))) - Array(abs.(fft(mean(tmpu .* tmp, dims=2))))
∇c = Array(abs.(fft(mean(s¹, dims=2))))
effective_diffusivities = (uc ./ ∇c)[index_choices]

kˣ_A = Array(kˣ)[index_choices]

# Array(abs.(fft(mean(tmp, dims=2))))
#=
fig = Figure()
ax = Axis(fig[1, 1])
scatter!(ax, kˣ_A, effective_diffusivities)
ylims!(0, 0.25)
=#

##
#=
using HDF5
filename = "effective_diffusivities_ensemble_correct_" * string(ensemble_size) * ".h5"
fid = h5open(filename, "w")
fid["effective_diffusivities"] = effective_diffusivities
fid["amplitude_factor"] = amplitude_factor
fid["phase_factor"] = 1.0
fid["ensemble_mean"] = tmp
fid["ensembe_flux_u"] = tmpuc
fid["ensembe_flux_v"] = tmpvc
close(fid)
=#


