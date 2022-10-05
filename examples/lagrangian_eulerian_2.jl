using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2, GLMakie, HDF5
using ProgressBars
rng = MersenneTwister(1234)
Random.seed!(123456789)
# jld_name = "high_order_timestep_spatial_tracer_"
jld_name = "default_streamfunction_"
include("transform.jl")
include("random_phase_kernel.jl")
# using GLMakie
save_fields = false
using CUDA
arraytype = CuArray
Ω = S¹(4π)^2
N = 2^7 # number of gridpoints

# for (di, amplitude_factor) in ProgressBar(enumerate([0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]))
di = 1
amplitude_factor = 0.5

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
maxind = minimum([40, floor(Int, N/4)])
index_choices = 2:maxind
tic = Base.time()

s .*= 0.0
for index_choice in ProgressBar(index_choices)
    kᶠ = kˣ[index_choice]
    @. θ = cos(kᶠ * x) / (kᶠ)^2 / κ # scaling so that source is order 1
    P * θ # in place fft
    @. κΔθ = κ * Δ * θ
    P⁻¹ * κΔθ # in place fft
    s .+= -κΔθ # add to source
end

t = [0.0]
tend = 30 # 5000

iend = ceil(Int, tend / Δt)

# simulation_parameters = (; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, s, P, P⁻¹, filter)
s .*= false
simulation_parameters = (; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, filter) 

φ .= arraytype(2π * rand(size(A)...))
θ .*= 0.0
θ̅ .*= 0.0
size_of_A = size(A)

rhs! = θ_rhs_symmetric_zeroth!

println("starting simulations")
scale = 1.0
tendish = floor(Int, tend / (scale * Δt))
tlist = cumsum(ones(tendish) .* Δt * scale)
indlist = [ceil(Int, t / Δt) for t in tlist]
indlist[1] = 1
lagrangian_list = zeros(length(indlist))
eulerian_list = copy(lagrangian_list)

realizations = 10000
tmpA = []
for j in ProgressBar(1:realizations)

    # new realization of flow
    rand!(rng, φ)
    φ .*= 2π
    event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
    wait(event)

    # get horizontal velocity
    P * ψ
    u0 = ∂y .* ψ
    P⁻¹ * ψ
    P⁻¹ * u0

    # initialize tracer
    θ .= u0
    if (j == 1) | (j == realizations)
        push!(tmpA, real.(Array(u)))
    end

    t[1] = 0.0

    ii = 1
    for i = 1:iend
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

        if i in indlist
            # get horizontal velocity
            P * ψ
            @. u = ∂y * ψ
            P⁻¹ * ψ
            P⁻¹ * u

            lagrangian_list[ii] += real(mean(θ .* u0)) / realizations
            eulerian_list[ii] += real(mean(u .* u0)) / realizations

            ii += 1
            # println("saveing at ", i)
        end

    end
    # println("finished realization ", j)
    @. θ̅ += θ / realizations
end

toc = Base.time()
println("the time for the simulation was ", toc - tic, " seconds")

fig = Figure()
ax = Axis(fig[1, 1]; xlabel="log10(time)", ylabel="autocorrelation", xlabelsize=30, ylabelsize=30)

logtlist = log10.(tlist)
ln1 = lines!(ax, logtlist[2:end], lagrangian_list[2:end], color=:blue, label="Lagrangian")
ln2 = lines!(ax, logtlist[2:end], eulerian_list[2:end], color=:orange, label="Eulerian")
axislegend(ax, position=:rt)
display(fig)


fig = Figure()
ax = Axis(fig[1, 1]; xlabel="time", ylabel="autocorrelation", xlabelsize=30, ylabelsize=30)
ylims!(ax, (-0.1 * amplitude_factor^2, amplitude_factor^2))

ln1 = lines!(ax, tlist[2:end], lagrangian_list[1:end-1], color=:blue, label="Lagrangian")
ln2 = lines!(ax, tlist[2:end], eulerian_list[2:end], color=:orange, label="Eulerian")
# the factor of 12 comes from the sqrt(1/12) factor in the random phase definition and the 1/2 comes from 
# fokker-planck nonsense of factors of two
ln3 = lines!(ax, tlist[2:end], eulerian_list[1] .* exp.(-1.0 .* tlist[1:end-1]), color=:red, label="Eulerian Analytic")
axislegend(ax, position=:rt)
display(fig)
