using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, HDF5
# using GLMakie, HDF5
using ProgressBars
rng = MersenneTwister(1234)
Random.seed!(123456789)
# grab computational kernels: functions defined
include("random_phase_kernel.jl")
# initialize fields: variables and domain defined here
include("initialize_fields.jl")

save_fields = true
if save_fields
    filename = "quick_effective_diffusivities" * ".hdf5"
    fid = h5open(filename, "w")
    create_group(fid, "effective_diffusivities")
    create_group(fid, "amplitude_factor")
    create_group(fid, "phase_speed")
    create_group(fid, "kernel_estimate")
    create_group(fid, "kappa")
end


# for λ ∈ ProgressBar([0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
amplitude_factors = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 32.0]
for (di, amplitude_factor) in ProgressBar(enumerate(amplitude_factors))

λ = 0
phase_speed = 1.0 # sqrt(1.0 * 0.04) # makes the decorrelation time 1ish

A .= default_A * amplitude_factor
##
# κ = 2 / N  # roughly 1/N for this flow
# κ = 2 / 2^8 # fixed diffusivity
# κ = 2e-4
Δx = x[2] - x[1]
κ = 5e-3 * amplitude_factor 
cfl = 0.1
Δx = (x[2] - x[1])
advective_Δt = cfl * Δx / amplitude_factor * 0.5
diffusive_Δt = cfl * Δx^2 / κ
Δt = minimum([advective_Δt, diffusive_Δt])

# take the initial condition as negative of the source
maxind = minimum([40, floor(Int, N[1] / 4)])
index_choices = 2:maxind
tic = Base.time()

tstart = 100 / amplitude_factor
s .*= 0.0
for index_choice in ProgressBar(index_choices)
    kᶠ = kˣ[index_choice]
    @. θ = cos(kᶠ * x) / (kᶠ)^2 / κ # scaling so that source is order 1
    P * θ # in place fft
    @. κΔθ = κ * Δ * θ
    P⁻¹ * κΔθ # in place fft
    s .+= -κΔθ # add to source
end

@. Δ = Δ - λ / κ # modify with λ

t = [0.0]
tend = 1000 / amplitude_factor

iend = ceil(Int, tend / Δt)

simulation_parameters = (; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ)

φ .= arraytype(2π * rand(size(A)...))
θ .*= 0.0
θ̅ .*= 0.0
size_of_A = size(A)

rhs! = θ_rhs_symmetric!

for i = ProgressBar(1:iend)
    # fourth order runge kutta on deterministic part
    # keep ψ frozen is the correct way to do it here

    # the below assumes that φ is just a function of time
    rhs!(k₁, θ, simulation_parameters)
    @. θ̃ = θ + Δt * k₁ * 0.5

    φ_rhs_normal!(φ̇, φ, rng)

    @. φ += phase_speed * sqrt(Δt / 2) * φ̇

    rhs!(k₂, θ̃, simulation_parameters)
    @. θ̃ = θ + Δt * k₂ * 0.5
    rhs!(k₃, θ̃, simulation_parameters)
    @. θ̃ = θ + Δt * k₃

    φ_rhs_normal!(φ̇, φ, rng)
    @. φ += phase_speed * sqrt(Δt / 2) * φ̇

    rhs!(k₄, θ̃, simulation_parameters)
    @. θ += Δt / 6 * (k₁ + 2 * k₂ + 2 * k₃ + k₄)

    t[1] += Δt
    # save output

    if t[1] >= tstart
        θ̅ .+= Δt * θ
    end

end

θ̅ ./= (t[end] - tstart)

toc = Base.time()
# println("the time for the simulation was ", toc - tic, " seconds")
# println("saving ", jld_name * string(index_choice) * ".jld2")
# factor of 2 comes the scaling factor of the fft (N[1]/2) 
tmp = Array(real.(fft(mean(θ̅, dims=2)[:])))
kxa = Array(kˣ)[:]
effective_diffusivities = (( (N[1]/2) ./ tmp) .- λ) ./ (kxa .^ 2) .- κ
effective_diffusivities = effective_diffusivities[index_choices]

# estimate kernel on grid
kernel = real.(fft([0.0, effective_diffusivities..., zeros(65)..., reverse(effective_diffusivities)...]))
kernel = kernel .- mean(kernel[63:65])
kernel = circshift(kernel, 64)

if save_fields
    fid["effective_diffusivities"][string(di)] = effective_diffusivities
    fid["amplitude_factor"][string(di)] = amplitude_factor
    fid["phase_speed"][string(di)] = phase_speed
    fid["kernel_estimate"][string(di)] = kernel
    fid["kappa"][string(di)] = κ
end


end
close(fid)
