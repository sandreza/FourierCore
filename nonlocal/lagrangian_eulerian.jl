using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2
# using GLMakie, HDF5
using ProgressBars
rng = MersenneTwister(1234)
Random.seed!(123456789)
# grab computational kernels: functions defined
include("random_phase_kernel.jl")
# initialize fields: variables and domain defined here
include("initialize_fields.jl")

# set amplitude
amplitude_factor = 1.0 # normalized later
list1 = Vector{Float64}[]
list2 = Vector{Float64}[]
# [sqrt(0.01), sqrt(0.05), sqrt(0.1), sqrt(0.2), sqrt(0.35), sqrt(0.5), sqrt(1.0)]
phase_speeds = [sqrt(0.01), sqrt(0.05), sqrt(0.1), sqrt(0.2), sqrt(0.35), sqrt(0.5), sqrt(1.0)]
for phase_speed ∈ ProgressBar(phase_speeds)
# phase_speed = sqrt(0.2) # sqrt(1.0 * 0.04) # makes the decorrelation time 1ish

# set the amplitude for the velocity field
A .= default_A * amplitude_factor

# set diffusivity and timestep
Δx = x[2] - x[1]
κ = 5e-3 * amplitude_factor # 0.01 * (2^7 / N[1])^2# amplitude_factor * 2 * Δx^2
cfl = 0.1 * 10
Δx = (x[2] - x[1])
advective_Δt = cfl * Δx / amplitude_factor * 0.5
diffusive_Δt = cfl * Δx^2 / κ
Δt = minimum([advective_Δt, diffusive_Δt])

# make source term zero
s .*= false

t = [0.0]
tend = minimum([1.0 / amplitude_factor, 1.0]) .* 10 # might want to make this a function of γ, κ, amplitude_factor, etc.
iend = ceil(Int, tend / Δt)

# all of these are defined in initialize_fields.jl
simulation_parameters = (; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ)

φ .= arraytype(2π * rand(size(A)...))

rhs! = θ_rhs_symmetric!

println("starting simulations")
scale = 1.0
tendish = floor(Int, tend / (scale * Δt))
tlist = cumsum(ones(tendish) .* Δt * scale)
indlist = [ceil(Int, t / Δt) for t in tlist]
indlist[1] = 1
# lagrangian_list = zeros(N..., length(indlist))
lagrangian_list = zeros(N..., length(1:iend+1))
lagrangian_list_2 = copy(lagrangian_list)
eulerian_list = copy(lagrangian_list)
theta_list = copy(lagrangian_list)
u_list = copy(lagrangian_list)

realizations = 1000
tmpA = []

for j in ProgressBar(1:realizations)

    # new realization of flow
    rand!(rng, φ)
    φ .*= 2π
    event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
    wait(event)

    # get horizontal velocity
    P * ψ
    u0 = -∂y .* ψ # typo here
    P⁻¹ * ψ
    P⁻¹ * u0

    # initialize tracer
    θ .= u0
    if (j == 1) | (j == realizations)
        push!(tmpA, real.(Array(u)))
    end

    t[1] = 0.0

    lagrangian_list[:, :, 1] .+= Array(real.(θ .* u0) ./ realizations)
    lagrangian_list_2[:, :, 1] .+= Array(real.(θ .* u0) ./ realizations)
    eulerian_list[:, :, 1] .+= Array(real.(u0 .* u0) ./ realizations)
    theta_list[:, :, 1] .+= Array(real.(θ) ./ realizations)
    u_list[:, :, 1] .+= Array(real.(u0) ./ realizations)

    ii = 1
    for i = 1:iend
        # fourth order runge kutta on deterministic part
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
        @. θ += Δt / 6 * real(k₁ + 2 * k₂ + 2 * k₃ + k₄)
        @. θ = real(θ)

        t[1] += Δt

        P * ψ
        @. u = -∂y * ψ
        P⁻¹ * ψ
        P⁻¹ * u

        lagrangian_list[:, :, i+1] .+= Array(real.(θ .* u0) ./ realizations)
        lagrangian_list_2[:, :, i+1] .+= Array(real.(θ .* u) ./ realizations)
        eulerian_list[:, :, i+1] .+= Array(real.(u .* u0) ./ realizations)
        theta_list[:, :, i+1] .+= Array(real.(θ) ./ realizations)
        u_list[:, :, i+1] .+= Array(real.(u) ./ realizations)
    end
    @. θ̅ += θ / realizations
end

llist = mean(lagrangian_list, dims=(1, 2))[1:end]
llist2 = mean(lagrangian_list_2, dims=(1, 2))[1:end]
elist = mean(eulerian_list, dims=(1, 2))[1:end]

tlist = [0, tlist..., tlist[end] + Δt]
push!(list1, llist2)
push!(list2, elist)
end

filename = "lagrangian_vs_eulerian.hdf5"
fid = h5open(filename, "w")
for i in eachindex(list2)
    fid["eulerian index " * string(i)] = list2[i]
    fid["lagrangian index " * string(i)] = list1[i]
end
fid["phase speeds"] = phase_speeds
fid["time"] = tlist
close(fid)

#=
using GLMakie

fig2 = Figure()
ax = Axis(fig2[1, 1]; xlabel="time", ylabel="autocorrelation", xlabelsize=30, ylabelsize=30)
# ylims!(ax, (minimum(llist2), 1.1 * elist[1]))

# ln1 = lines!(ax, tlist[1:end], llist, color=:blue, label="Lagrangian (incorrect)")
ln2 = lines!(ax, tlist[1:end], elist, color=:orange, label="Eulerian")

# ln3 = lines!(ax, tlist[1:end], elist[1] .* exp.(-1.0 .* tlist), color=:red, label="Eulerian Analytic")
ln4 = lines!(ax, tlist[1:end], llist2, color=:purple, label="Lagrangian")
ylims!(ax, (-1.1 * elist[1], 1.1 * elist[1]))
axislegend(ax, position=:rt)
display(fig2)

println("The lagrangian estimate is ", sum(llist2 .* Δt))
println("The Eulerian estimate is ", sum(elist .* Δt))
println("Molecular value", κ)
println("ratio of lagrangian to Eulerian is ", sum(llist2 .* Δt) / sum(elist .* Δt))
println("ratio of lagrangian to molecular is ", sum(llist2 .* Δt) / κ)


##
fig = Figure() 
ax11 = Axis(fig[1, 1]; xlabel="x", ylabel = "y", title="u", xlabelsize=30, ylabelsize=30)
ax12 = Axis(fig[1, 2]; xlabel="x", ylabel = "y", title="v", xlabelsize=30, ylabelsize=30)
ax21 = Axis(fig[2, 1]; xlabel="x", ylabel = "y", title="θ", xlabelsize=30, ylabelsize=30)
ax22 = Axis(fig[2, 2]; xlabel="x", ylabel = "y", title="ψ", xlabelsize=30, ylabelsize=30)
x_A = Array(x)[:] 
y_A = Array(y)[:] 
u_A = real.(Array(u))
v_A = real.(Array(v))
θ_A = real.(Array(θ))
ψ_A = real.(Array(ψ))
scale = 60
heatmap!(ax11, x_A, y_A, u_A, colorrange=(-1, 1) .* scale, colormap = :balance)
heatmap!(ax12, x_A, y_A, v_A, colorrange=(-1, 1) .* scale, colormap = :balance)
heatmap!(ax21, x_A, y_A, θ_A, colorrange=(-1, 1) .* scale, colormap = :balance)
heatmap!(ax22, x_A, y_A, ψ_A, colorrange=(-1, 1) .* scale, colormap = :balance)
=#
