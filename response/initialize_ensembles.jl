include("timestepping.jl") 
include("initialize_fields.jl") 
include("default_constants.jl")
include("initialize_operators.jl")

parameters = (; auxiliary, operators, constants) 
# time
t = [0.0]  
tend = 2^18

for i in ProgressBar(1:tend)
    t[1] += Δt
    step!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, t, parameters)
end
ζ1 =  copy(ζ)
##
# time
t = [0.0]  
tend = 2^8
include("initialize_fields.jl") 
ζs = Array{ComplexF64, 3}[]
ζ .= ζ1
for i in ProgressBar(1:tend)
    push!(ζs, Array(copy(ζ)))
    t[1] += Δt
    step!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, t, parameters)
end

ζ2 =  copy(ζ)

include("initialize_fields.jl") 
ζs_δ = Array{ComplexF64, 3}[]
ζ .= ζ1
δ = 10 * Δt
ζ[1, 1, :] .+= δ
for i in ProgressBar(1:tend)
    push!(ζs_δ, Array(copy(ζ)))
    t[1] += Δt
    step!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, t, parameters)
end

ζ3 =  copy(ζ)
##
Rᵢⱼ =  [mean(real.((ζs_δ[i] - ζs[i] ) / δ), dims = 3)[:,:,1] for i in eachindex(ζs)]
R₁₁ = [Rᵢⱼ[i][1,1] for i in eachindex(Rᵢⱼ)]
Rmax = [maximum(Rᵢⱼ[i]) for i in eachindex(Rᵢⱼ)]
maximum(Rᵢⱼ[end])
##
extrema(real.(ζ)[:,:,1])
using GLMakie
heatmap(real.(ζ)[:,:,1], colormap = :balance, colorrange = (-2,2), interpolate =true)