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
defined outside
=#

maxind = minimum([40, floor(Int, N[1] / 2)])
index_choices = 2:maxind

#= 
forcing_amplitude = 300.0
ϵ = 1.0
ω = 2π / 2^5
constants = (; forcing_amplitude=forcing_amplitude, ϵ=ϵ, ω=ω)
parameters = (; auxiliary, operators, constants)

=# 

load_psi!(ψ)
P * ψ;
ζ .= Δ .* ψ
P⁻¹ * ζ
@. u = (∂y * ψ)
P⁻¹ * u
θ .= u

start_index = floor(Int, tstart / Δt)
sθ .= 0.0
θ .= 0
t = [0.0]
iend = ceil(Int, tend / Δt)
# new realization of flow
rand!(rng, φ) # between 0, 1
φ .*= 2π # to make it a random phase

iter = ProgressBar(1:iend)
eke_list = Float64[]
uθ_list = Float64[]

push!(uθ_list, real(mean(θ .* u)))
push!(eke_list, real(mean(u .* u + v .* v)))
t .= 0.0
for i = iter
    step!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, t, parameters)
    push!(uθ_list, real(mean(θ .* u)))
    if i % 10 == 0
        θ_min, θ_max = extrema(real.(θ))
        ζ_min, ζ_max = extrema(real.(ζ))
        s1 = "θ_min: $θ_min \nθ_max: $θ_max \nζ_min: $ζ_min \nζ_max: $ζ_max"
        s2 = "\nuθ   : $(uθ_list[i])"
        set_multiline_postfix(iter, s1 * s2)
    end
    if i % 10 == 0
        push!(eke_list, real(mean(u .* u + v .* v)))
    end
end
