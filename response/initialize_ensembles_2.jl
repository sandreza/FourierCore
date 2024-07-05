include("timestepping.jl") 
include("initialize_fields.jl") 
include("default_constants.jl")
include("initialize_operators.jl")

Nemax = 128
parameters = (; auxiliary, operators, constants) 
# time
t = [0.0]  
tend = 2^17

for i in ProgressBar(1:tend)
    t[1] += Δt
    step!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, t, parameters)
end
ζ1 =  copy(ζ)
##
# time
t = [0.0]  
tend = 2^10
include("initialize_fields.jl") 
ζs = [zeros(Ns[1], Ns[2],  Nemax) for i in 1:tend]
snapshot_inds = collect(1:256:tend)
snapshots = zeros(Float32, Ns[1], Ns[2], Ns[3], length(snapshot_inds))

ζ .= ζ1
j = [0]
for i in ProgressBar(1:tend)
    ζs[i] .= real.(Array(copy(ζ)))[:,:, 1:Nemax]
    if i in snapshot_inds
        j .+= 1
        snapshots[:, :, :, j[1]] .= Float32.(real.(Array(copy(ζ)))[:, :, :])
    end
    t[1] += Δt
    step!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, t, parameters)
end

ζ2 =  copy(ζ)

include("initialize_fields.jl") 
ζs_δ = [zeros(Ns[1], Ns[2],  Nemax) for i in 1:tend]
ζ .= ζ1
δ = 2 * Δt
ζ[1, 1, :] .+= δ
for i in ProgressBar(1:tend)
    ζs_δ[i] .= real.(Array(copy(ζ)))[:,:, 1:Nemax]
    t[1] += Δt
    step!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, t, parameters)
end

ζ3 =  copy(ζ)
##
@info "computing perturbations"
Rᵢⱼ =  [mean(real.((ζs_δ[i] - ζs[i] ) / δ), dims = 3)[:,:,1] for i in eachindex(ζs)]
# R₁₁ = [Rᵢⱼ[i][1,1] for i in eachindex(Rᵢⱼ)]
# Rmax = [maximum(Rᵢⱼ[i]) for i in eachindex(Rᵢⱼ)]
# maximum(Rᵢⱼ[end])
#=
##
extrema(real.(ζ)[:,:,1])
# using GLMakie
# heatmap(real.(ζ)[:,:,1], colormap = :balance, colorrange = (-2,2), interpolate =true)
##
#=
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
=#

##
=#
@info "computing trajectories"
Nt = length(ζs)
# trj = zeros(Ns[1] * Ns[2], Nemax*Nt)
trj2 = zeros(Ns[1], Ns[2], Nemax, Nt)

for i in ProgressBar(1:Nt)
    # trj[:, (i-1)*Nemax .+ 1:i*Nemax] .= real.(reshape(ζs[i][:], (Ns[1] * Ns[2], Nemax)))
    trj2[:, :, :, i] .= real.(ζs[i])
end

@info "computing linear response"
include("utils.jl")
rf = linear_response_function(trj2; skip = 1)
Rᵢⱼ_linear = [reshape(rf[i][:, 1], (Ns[1], Ns[2])) for i in eachindex(rf)]
R₁₁_linear = [rf[i][1,1] for i in eachindex(rf)]

Rij_dynamic = zeros(Ns[1], Ns[2], Nt÷2)
Rij_linear = zeros(Ns[1], Ns[2], Nt÷2)
for i in ProgressBar(1:Nt÷2)
    Rij_dynamic[:, :, i] .= Rᵢⱼ[i]
    Rij_linear[:, :, i] .= Rᵢⱼ_linear[i]
end
##
#=
fig = Figure()
for i in 1:16
    ii = (i-1)÷4 + 1
    jj = (i-1)%4 + 1
    ax = Axis(fig[ii, jj]; title = "Response $i, 1")
    scatter!(ax, Rij_dynamic[i, 1, :], label = "Dynamic", color = :blue)
    lines!(ax, Rij_linear[i, 1, :], label = "Linear", color = :orange)
    if i == 1
        axislegend(ax, position = :rt, labelsize = 30)
    end
    xlims!(ax, 0, 400)
end
display(fig)
save("response1.png", fig)
=#
##
#=
fig2 = Figure() 
for i in 1:16
    ii = (i-1)÷4 + 1
    jj = (i-1)%4 + 1
    tind = (i-1) * 64 + 1
    ax = Axis(fig2[ii, jj]; title = "time $tind ")
    heatmap!(ax, real.(ζs[tind][:,:,1]), colormap = :balance, colorrange = (-2,2), interpolate =true)
end
display(fig2)
=#
##
##
@info "timeseries"
scale_factor = 5
r_snapshots = reshape(snapshots,  (Ns[1], Ns[2], 1, Ns[3] * length(snapshot_inds)) ) / scale_factor
timeseries = reshape(trj2, (Ns[1], Ns[2], 1,  Nemax * size(trj2)[end])) / scale_factor
a, b, c, d = size(r_snapshots)
ilist = rand(1:d, 100)
jlist = rand(1:d, 100)
σ = maximum([norm(r_snapshots[:, :, 1, i][:] - r_snapshots[:, :, 1, j][:]) for i in ilist, j in jlist]) * 1.2
##
using HDF5
hfile = h5open("/nobackup1/sandre/ResponseFunctionTrainingData/data_7.0_0.0_0.0_0.0.hdf5", "w")
hfile["snapshots"] = Float32.(r_snapshots)
hfile["timeseries"] = Float32.(timeseries)
hfile["snapshots shape"]  =  [Ns[1], Ns[2], Ns[3], length(snapshot_inds)]
hfile["timeseries shape"] =  [Ns[1], Ns[2], 1, Nemax, tend]
hfile["sigma"] = σ
hfile["scale factor"] = scale_factor
close(hfile)

hfile = h5open("/nobackup1/sandre/ResponseFunctionTrainingData/data_7.0_0.0_0.0_0.0_responses.hdf5", "w")
hfile["linear response"] = Rij_linear
hfile["perturbation response"] = Rij_dynamic
close(hfile)
