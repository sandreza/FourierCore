include("timestepping.jl") 
include("initialize_fields.jl") 
include("default_constants.jl")
include("initialize_operators.jl")

parameters = (; auxiliary, operators, constants) 
# time
t = [0.0]  
tend = 2^16

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
δ = 2 * Δt
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
##

Nplot = 5
tinds = round.(Int, range(1, tend, length = Nplot^2))
fig = Figure()
for i in 1:Nplot^2
    ii = (i-1)÷Nplot + 1
    jj = (i-1)%Nplot + 1
    ax = Axis(fig[ii,jj])
    heatmap!(ax, Rᵢⱼ[tinds[i]], colormap = :balance, colorrange = (-1,1), interpolate = true)
end
display(fig)


##
Nt = length(ζs)
trj = zeros(Ns[1] * Ns[2], Ns[3]*Nt)
trj2 = zeros(Ns[1], Ns[2], Ns[3], Nt)
for i in ProgressBar(1:Nt)
    trj[:, (i-1)*Ns[3] .+ 1:i*Ns[3]] .= real.(reshape(ζs[i][:], (Ns[1] * Ns[2], Ns[3])))
    trj2[:, :, :, i] .= real.(ζs[i])
end

include("utils.jl")


rf = linear_response_function(trj2; skip = 1)
R₁₁_linear = [rf[i][1,1] for i in eachindex(rf)]
##
fig = Figure()
ax = Axis(fig[1,1])
scatter!(ax, R₁₁, label = "Dynamic", color = :blue)
lines!(ax, R₁₁_linear, label = "Linear", color = :orange)
axislegend(ax, position = :rt, labelsize = 10)
display(fig)
