@info "diffusivity kernel"
using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2
# using GLMakie, HDF5
using ProgressBars
rng = MersenneTwister(1234)
Random.seed!(123456789)

maxind = minimum([40, floor(Int, N[1] / 2)])
index_choices = 2:maxind

start_index = floor(Int, tstart / Δt)

sθ .= 0.0
θ .= 0
index_choices_1 = 2:2
index_choices_2 = 3:3
for index_choice_1 in ProgressBar(index_choices_1)
    for index_choice_2 in ProgressBar(index_choices_2)
        k¹ = kˣ[index_choice_1]
        k² = kʸ[index_choice_2]
        @. θ += cos(k¹ * x) * cos(k² * y) /((k¹)^2 + (k¹)^2) / κ # scaling so that source is order 1
    end
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

θ̄ = arraytype(zeros(ComplexF64, N, N))
uθ_average = arraytype(zeros(ComplexF64, N, N))
vθ_average = arraytype(zeros(ComplexF64, N, N))
∂ˣθ_average = arraytype(zeros(ComplexF64, N, N))
∂ʸθ_average = arraytype(zeros(ComplexF64, N, N))

iter = ProgressBar(1:iend)
ke_list = Float64[]
t .= 0.0
for i = iter
    step!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, t, parameters)
    if i % mod_index == 0
        θ_min, θ_max = extrema(real.(θ))
        ζ_min, ζ_max = extrema(real.(ζ))
        set_multiline_postfix(iter, "θ_min: $θ_min \nθ_max: $θ_max \nζ_min: $ζ_min \nζ_max: $ζ_max")
    end
    if i > start_index
        θ̄ .+= Δt .* mean(θ, dims = 3)
        uθ_average .+= Δt .* mean(u .* θ, dims = 3)
        vθ_average .+= Δt .* mean(v .* θ, dims = 3)
        ∂ˣθ_average .+= Δt .* mean(∂ˣθ, dims = 3)
        ∂ʸθ_average .+= Δt .* mean(∂ʸθ, dims = 3)
    end
    if i % mod_index == 0
        push!(ke_list, real(mean(u .* u + v .* v)))
    end
end

θ̄ ./= (tend - tstart)
uθ_average ./= (tend - tstart)
vθ_average ./= (tend - tstart)
∂ˣθ_average ./= (tend - tstart)
∂ʸθ_average ./= (tend - tstart)
θ̄_A = Array(real.(θ̄))
uθ_average_A = Array(real.(uθ_average))
vθ_average_A = Array(real.(vθ_average))
∂ˣθ_average_A = Array(real.(∂ˣθ_average))
∂ʸθ_average_A = Array(real.(∂ʸθ_average))



@info "done with diffusivity kernel"

#=
start_index = floor(Int, tstart / Δt)

sθ .= 0.0
θ .= 0

for index_choice_1 in ProgressBar(index_choices)
    for index_choice_2 in ProgressBar(index_choices)
        k¹ = kˣ[index_choice_1]
        k² = kʸ[index_choice_2]
        @. θ += sin(k¹ * x) * cos(k² * y) /((k¹)^2 + (k¹)^2) / κ # scaling so that source is order 1
    end
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

θ̄ = arraytype(zeros(ComplexF64, N, N))
uθ_average = arraytype(zeros(ComplexF64, N, N))
vθ_average = arraytype(zeros(ComplexF64, N, N))
∂ˣθ_average = arraytype(zeros(ComplexF64, N, N))
∂ʸθ_average = arraytype(zeros(ComplexF64, N, N))

iter = ProgressBar(1:iend)
ke_list = Float64[]
t .= 0.0
for i = iter
    step!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, t, parameters)
    if i % mod_index == 0
        θ_min, θ_max = extrema(real.(θ))
        ζ_min, ζ_max = extrema(real.(ζ))
        set_multiline_postfix(iter, "θ_min: $θ_min \nθ_max: $θ_max \nζ_min: $ζ_min \nζ_max: $ζ_max")
    end
    if i > start_index
        θ̄ .+= Δt .* mean(θ, dims = 3)
        uθ_average .+= Δt .* mean(u .* θ, dims = 3)
        vθ_average .+= Δt .* mean(v .* θ, dims = 3)
        ∂ˣθ_average .+= Δt .* mean(∂ˣθ, dims = 3)
        ∂ʸθ_average .+= Δt .* mean(∂ʸθ, dims = 3)
    end
    if i % mod_index == 0
        push!(ke_list, real(mean(u .* u + v .* v)))
    end
end

θ̄ ./= (tend - tstart)
uθ_average ./= (tend - tstart)
vθ_average ./= (tend - tstart)
∂ˣθ_average ./= (tend - tstart)
∂ʸθ_average ./= (tend - tstart)
θ̄_A = Array(real.(θ̄))
uθ_average = Array(real.(uθ_average))
vθ_average = Array(real.(vθ_average))
∂ˣθ_average = Array(real.(∂ˣθ_average))
∂ʸθ_average = Array(real.(∂ʸθ_average))


tmpsave = Array(real.(mean(θ̄, dims=2)))[:, 1, :]
tmp = Array(real.(fft(mean(θ̄, dims=(2, 3))[:]))) # tmp = real.(fft(Array(mean(θ[:,:,1:10], dims = (2,3)))[:]))
kxa = Array(kˣ)[:]
effective_diffusivities = ((Ns[1] / 2) ./ tmp) ./ (kxa .^ 2) .- κ # (((N[1] / 2) ./ tmp) .- λ) ./ (kxa .^ 2) .- κ
effective_diffusivities = effective_diffusivities[index_choices]

# estimate kernel on grid
kernel = real.(fft([0.0, effective_diffusivities..., zeros(65)..., reverse(effective_diffusivities)...]))
kernel = kernel .- mean(kernel[63:65])
kernel = circshift(kernel, 64)

@info "done with diffusivity kernel"
=#