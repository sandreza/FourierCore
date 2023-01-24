using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2
# using GLMakie, HDF5
using ProgressBars
rng = MersenneTwister(1234)
Random.seed!(123456789)

# initialize fields: variables and domain defined here
include("initialize_fields.jl")

filename = "quick_effective_diffusivities" * ".hdf5"
fid = h5open(filename, "w")

ϵ = 1.0
tstart = 100.0
tend = 300.0

for (ii, ϵ) ∈ ProgressBar(enumerate([0.01, 0.1, 1.0]))
    constants = (; forcing_amplitude=forcing_amplitude, ϵ=ϵ)
    parameters = (; auxiliary, operators, constants)



    start_index = floor(Int, tstart / Δt)
    amplitude_factor = 1.0 # normalized later
    phase_speed = 1.0
    λ = 0
    sθ .= 0.0
    θ .= 0
    maxind = minimum([40, floor(Int, N[1] / 3)])
    index_choices = 2:maxind
    for index_choice in ProgressBar(index_choices)
        kᶠ = kˣ[index_choice]
        @. θ += cos(kᶠ * x) / (kᶠ)^2 / κ # scaling so that source is order 1
    end
    P * θ # in place fft
    @. 𝒟θ = 𝒟κ * θ
    P⁻¹ * 𝒟θ # in place fft
    P⁻¹ * θ # in place fft
    sθ .+= -𝒟θ # add to source

    t = [0.0]
    iend = ceil(Int, tend / Δt)

    # new realization of flow
    rand!(rng, φ) # between 0, 1
    φ .*= 2π # to make it a random phase

    load_psi!(ψ)
    ζ .= ifft(Δ .* fft(ψ))

    θ̄ = arraytype(zeros(ComplexF64, N, N))

    iter = ProgressBar(1:iend)
    for i = iter
        step!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, parameters)
        if i % 10 == 0
            θ_min, θ_max = extrema(real.(θ))
            ζ_min, ζ_max = extrema(real.(ζ))
            set_multiline_postfix(iter, "θ_min: $θ_min \nθ_max: $θ_max \nζ_min: $ζ_min \nζ_max: $ζ_max")
        end
        if i > start_index
            θ̄ .+= Δt .* mean(θ, dims=3)[:, :]
        end
    end

    θ̄ ./= (tend - tstart)
    θ̄_A = Array(real.(θ̄))


    tmp = Array(real.(fft(mean(θ̄, dims=2)[:])))
    kxa = Array(kˣ)[:]
    effective_diffusivities = (((N[1] / 2) ./ tmp) .- λ) ./ (kxa .^ 2) .- κ
    effective_diffusivities = effective_diffusivities[index_choices]

    # estimate kernel on grid
    kernel = real.(fft([0.0, effective_diffusivities..., zeros(65)..., reverse(effective_diffusivities)...]))
    kernel = kernel .- mean(kernel[63:65])
    kernel = circshift(kernel, 64)

    #=
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="x", ylabel="y", aspect=1)
    scatter!(ax, kernel)
    display(fig)
    =#


    fid["effective_diffusivities "*string(ii)] = effective_diffusivities
    fid["amplitude "*string(ii)] = ϵ
    fid["kernel "*string(ii)] = kernel

end


close(fid)
