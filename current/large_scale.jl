@info "large scale case"
using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2
# using GLMakie, HDF5
using ProgressBars
rng = MersenneTwister(1234)
Random.seed!(123456789)

#=
scaleit = 2^3
tstart = 2^5 * scaleit
tend = 2^6 * scaleit
defined outside script 
=#
maxind = minimum([40, floor(Int, N[1] / 2)])
index_choices = 2:maxind
#=
forcing_amplitude = 300.0
ϵ = 1.0
ω = 0.0
constants = (; forcing_amplitude=forcing_amplitude, ϵ=ϵ, ω=ω)
parameters = (; auxiliary, operators, constants)
defined outsdie script
=#

# initialize tracer with velocity field
@. u = (∂y * ψ)
P⁻¹ * u
θ .= u


start_index = floor(Int, tstart / Δt)
sθ .= 0.0
t = [0.0]
iend = ceil(Int, tend / Δt)
# new realization of flow
rand!(rng, φ) # between 0, 1
φ .*= 2π # to make it a random phase

# θ̄ = arraytype(zeros(ComplexF64, N, N, N_ens))

iter = ProgressBar(1:iend)
ke_list = Float64[]
uθ_list = Float64[]
tlist = Float64[]

push!(uθ_list, real(mean(θ .* u)))
push!(ke_list, real(mean(u .* u + v .* v)))
push!(tlist, t[1])
t .= 0.0
for i = iter
    step!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, t, parameters)
    push!(uθ_list, real(mean(θ .* u)))
    push!(tlist, t[1])
    if i % mod_index == 0
        θ_min, θ_max = extrema(real.(θ))
        ζ_min, ζ_max = extrema(real.(ζ))
        s1 = "θ_min: $θ_min \nθ_max: $θ_max \nζ_min: $ζ_min \nζ_max: $ζ_max"
        s2 = "\nuθ   : $(uθ_list[i])"
        set_multiline_postfix(iter, s1 * s2)
    end
    if i > start_index
        # u is already in real space as is θ
        # θ̄ .+= Δt .* θ .* u
    end
    if i % mod_index == 0
        push!(ke_list, real(mean(u .* u + v .* v)))
    end
end

# θ̄ ./= (tend - tstart)
# θ̄_A = Array(real.(θ̄))

@info "done with large scale case"