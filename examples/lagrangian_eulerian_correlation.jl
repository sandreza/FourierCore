using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2
using GLMakie
rng = MersenneTwister(1234)
# Random.seed!(123456789)
Random.seed!(12)
# jld_name = "high_order_timestep_spatial_tracer_"
jld_name = "blocky"
include("transform.jl")
include("random_phase_kernel.jl")
# using GLMakie
using CUDA
arraytype = CuArray
Ω = S¹(4π)^2
N = 2^7 # number of gridpoints
grid = FourierGrid(N, Ω, arraytype=arraytype)
nodes, wavenumbers = grid.nodes, grid.wavenumbers

x = nodes[1]
y = nodes[2]
kˣ = wavenumbers[1]
kʸ = wavenumbers[2]
# construct filter
kxmax = maximum(kˣ)
kymax = maximum(kˣ)
filter = @. (kˣ)^2 + (kʸ)^2 ≤ ((kxmax / 2)^2 + (kymax / 2)^2)

# now define the random field 
phase_speed = 0.2 # default is 1.0
wavemax = 3
𝓀 = arraytype(collect(-wavemax:0.5:wavemax))
𝓀ˣ = reshape(𝓀, (length(𝓀), 1))
𝓀ʸ = reshape(𝓀, (1, length(𝓀)))
inertial_exponent = -3.0
stream_function_exponent = (inertial_exponent - 1) / 4;
A = @. 1e-0 * 0.2 * (𝓀ˣ * 𝓀ˣ + 𝓀ʸ * 𝓀ʸ)^(stream_function_exponent)
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

# number of gridpoints in transition is about λ * N / 2
bump(x; λ=10 / N, width=π / 2) = 0.5 * (tanh((x + width / 2) / λ) - tanh((x - width / 2) / λ))
bumps(x; λ=20 / N, width=1.0) = 0.25 * (bump(x, λ=λ, width=width) + bump(x, λ=λ, width=2.0 * width) + bump(x, λ=λ, width=3.0 * width) + bump(x, λ=λ, width=4.0 * width))

##
Δx = x[2] - x[1]
κ = 1.0 * Δx^2 # 1.0 / N * 2^(2)  # roughly 1/N for this flow
# κ = 2 / 2^8 # fixed diffusivity
# κ = 2e-4
Δt = (Δx) / (4π) * 5 

# take the initial condition as negative of the source
tic = Base.time()

# save some snapshots
ψ_save = typeof(real.(Array(ψ)))[]
θ_save = typeof(real.(Array(ψ)))[]

r_A = Array(@. sqrt((x - 2π)^2 + (y - 2π)^2))
θ_A = [bumps(r_A[i, j]) for i in 1:N, j in 1:N]
θ .= CuArray(θ_A)
# @. θ = bump(sqrt(x^2 + y^2)) # scaling so that source is order 1
θclims = extrema(Array(real.(θ))[:])
P * θ # in place fft
@. κΔθ = κ * Δ * θ
P⁻¹ * κΔθ # in place fft
s .= -κΔθ * 0.0
P⁻¹ * θ # in place fft
θ̅ .= 0.0

t = [0.0]
tend = 100 # 5000

iend = ceil(Int, tend / Δt)

scale = 1.0
tendish = floor(Int, tend / (scale * Δt))
tlist = cumsum(ones(tendish) .* Δt * scale)
indlist = [ceil(Int, t / Δt) for t in tlist]
indlist[1] = 1
lagrangian_list = zeros(length(indlist))
eulerian_list = copy(lagrangian_list)

params = (; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, s, P, P⁻¹, filter)

size_of_A = size(A)

realizations = 100
tmpA = []
for j in 1:realizations

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
        θ_rhs_new!(k₁, θ, params)
        @. θ̃ = θ + Δt * k₁ * 0.5

        φ_rhs_normal!(φ̇, φ, rng)
        @. φ += phase_speed * sqrt(Δt / 2) * φ̇

        θ_rhs_new!(k₂, θ̃, params)
        @. θ̃ = θ + Δt * k₂ * 0.5
        θ_rhs_new!(k₃, θ̃, params)
        @. θ̃ = θ + Δt * k₃

        φ_rhs_normal!(φ̇, φ, rng)
        @. φ += phase_speed * sqrt(Δt / 2) * φ̇

        θ_rhs_new!(k₄, θ̃, params)
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
    println("finished realization ", j)
    @. θ̅ += θ / realizations
end

toc = Base.time()
println("the time for the simulation was ", toc - tic, " seconds")

fig = Figure()
ax = Axis(fig[1,1]; xlabel = "log10(time)", ylabel = "autocorrelation", xlabelsize =30, ylabelsize =30)

logtlist = log10.(tlist)
ln1 = lines!(ax, logtlist[2:end], lagrangian_list[2:end], color=:blue, label = "Lagrangian")
ln2 = lines!(ax, logtlist[2:end], eulerian_list[2:end], color=:orange, label = "Eulerian")
axislegend(ax, position=:rc)
display(fig)

#=
x_A = Array(x)[:] .- 2π
θ_A = tmpA[1]
θ_F = tmpA[2]
θ̅_F = Array(real.(θ̅))

fig = Figure(resolution=(2048, 512))
ax1 = Axis(fig[1, 1], title="t = 0")
ax2 = Axis(fig[1, 2], title="instantaneous t = " * string(tend))
ax3 = Axis(fig[1, 4], title="ensemble average t = " * string(tend))
println("the extrema of the end field is ", extrema(θ_F))
println("the extrema of the ensemble average is ", extrema(θ̅_F))
colormap = :bone_1
colormap = :nipy_spectral
heatmap!(ax1, x_A, x_A, θ_A, colormap=colormap, colorrange=(0.0, 1.0), interpolate=true)
hm = heatmap!(ax2, x_A, x_A, θ_F, colormap=colormap, colorrange=(0.0, 1.0), interpolate=true)
hm_e = heatmap!(ax3, x_A, x_A, θ̅_F, colormap=colormap, colorrange=(0.0, 0.2), interpolate=true)
Colorbar(fig[1, 3], hm, height=Relative(3 / 4), width=25, ticklabelsize=30, labelsize=30, ticksize=25, tickalign=1,)
Colorbar(fig[1, 5], hm_e, height=Relative(3 / 4), width=25, ticklabelsize=30, labelsize=30, ticksize=25, tickalign=1,)
display(fig)
=#