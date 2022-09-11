using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2
using GLMakie, ProgressBars
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
phase_speed = 5
# eulerian decorrelation time is 24/phase_speed^2 
amplitude_factor = 2.0 # sqrt(phase_speed)
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
wavemax = 3
𝓀 = arraytype(collect(-wavemax:0.5:wavemax))
𝓀ˣ = reshape(𝓀, (length(𝓀), 1))
𝓀ʸ = reshape(𝓀, (1, length(𝓀)))
A = @. 0.1 * (𝓀ˣ * 𝓀ˣ + 𝓀ʸ * 𝓀ʸ)^(-11 / 12) * amplitude_factor
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
bump(x; λ=20 / N, width=π / 2 ) = 0.5 * (tanh((x + width / 2) / λ) - tanh((x - width / 2) / λ))
bumps(x; λ=20 / N, width=1.0) = 0.25 * (bump(x, λ=λ, width=width) + bump(x, λ=λ, width=2.0 * width) + bump(x, λ=λ, width=3.0 * width) + bump(x, λ=λ, width=4.0 * width))

##
Δx = x[2] - x[1]
κ = amplitude_factor * 2*Δx^2 
# κ = 2 / 2^8 # fixed diffusivity
# κ = 2e-4
Δt = (Δx) / (4π) * 5  / amplitude_factor 

# take the initial condition as negative of the source
tic = Base.time()

# save some snapshots
ψ_save = typeof(real.(Array(ψ)))[]
θ_save = typeof(real.(Array(ψ)))[]

r_A = Array(@. sqrt((x - 2π)^2 + (y - 2π)^2))
θ_A = [bump(r_A[i, j]) for i in 1:N, j in 1:N]
# θ_A = [bump.(abs.(x[i]- 2π)) for i in 1:N, j in 1:N]
θ .= CuArray(θ_A)
# @. θ = bump(sqrt(x^2 + y^2)) # scaling so that source is order 1
θclims = extrema(Array(real.(θ))[:])
P * θ # in place fft
@. κΔθ = κ * Δ * θ
P⁻¹ * κΔθ # in place fft
s .= -κΔθ 
P⁻¹ * θ # in place fft
θ̅ .= 0.0

t = [0.0]
tend = 5 

iend = ceil(Int, tend / Δt)

params = (; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, s, P, P⁻¹, filter)

size_of_A = size(A)

realizations = 10000

θ̅_timeseries = CuArray(zeros(size(ψ)..., iend))

for j in ProgressBar(1:realizations)
    # new realization of flow
    rand!(rng, φ) # between 0, 1
    φ .*= 2π # to make it a random phase
    event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
    wait(event)
    t[1] = 0

    θ .= CuArray(θ_A)
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

        # update stochastic part 
        # φ_rhs_normal!(φ̇, φ, rng)
        # @. φ += sqrt(Δt) * φ̇


        t[1] += Δt
        # save output
        # tmp = real.(Array(θ))
        @. θ̅_timeseries[:, :, i] += real.(θ) / realizations
        #=
        if i % div(iend, 10) == 0
            println("Saving at i=", i)
            push!(ψ_save, Array(real.(ψ)))
            push!(θ_save, Array(real.(θ)))
            println("extrema are ", extrema(θ_save[end]))
            println("time is t = ", t[1])
        end

        if t[1] >= tstart
            θ̅ .+= Δt * θ
        end

        if i % div(iend, 100) == 0
            println("time is t = ", t[1])
            local toc = Base.time()
            println("the time for the simulation is ", toc - tic, " seconds")
            println("extrema are ", extrema(real.(θ)))
            println("on wavenumber index ", index_choice)
        end
        =#

    end
    # println("finished realization ", j)
    @. θ̅ += θ / realizations
end

toc = Base.time()
println("the time for the simulation was ", toc - tic, " seconds")

x_A = Array(x)[:] .- 2π
θ_F = Array(real.(θ))
θ̅_F = Array(real.(θ̅))

begin
fig = Figure(resolution=(2048, 512))
ax1 = Axis(fig[1, 1], title="t = 0")
ax2 = Axis(fig[1, 2], title="instantaneous t = " * string(tend))
ax3 = Axis(fig[1, 4], title="ensemble average t = " * string(tend))
println("the extrema of the end field is ", extrema(θ_F))
println("the extrema of the ensemble average is ", extrema(θ̅_F))
colormap = :bone_1
# colormap = :nipy_spectral
heatmap!(ax1, x_A, x_A, θ_A, colormap=colormap, colorrange=(0.0, 1.0), interpolate=true)
hm = heatmap!(ax2, x_A, x_A, θ_F, colormap=colormap, colorrange=(0.0, 1.0), interpolate=true)
hm_e = heatmap!(ax3, x_A, x_A, θ̅_F, colormap=colormap, colorrange=(0.0, 0.2), interpolate=true)
Colorbar(fig[1, 3], hm, height=Relative(3 / 4), width=25, ticklabelsize=30, labelsize=30, ticksize=25, tickalign=1,)
Colorbar(fig[1, 5], hm_e, height=Relative(3 / 4), width=25, ticklabelsize=30, labelsize=30, ticksize=25, tickalign=1,)
display(fig)
end

##
begin
fig = Figure(resolution=(1400, 1100))
ax11 = Axis(fig[1, 1]; title="ensemble average")
ax12 = Axis(fig[1, 2]; title="x=0 slice")
ax21 = Axis(fig[2, 1]; title="diffusion")
ax22 = Axis(fig[2, 2]; title="nonlocal space kernel")
t_slider = Slider(fig[3, 1:2], range=1:iend, startvalue=0)
tindex = t_slider.value
colormap = :bone_1
field = @lift(Array(θ̅_timeseries[:, :, $tindex]))
field_slice = @lift($field[:, floor(Int, N / 2)]) 

# particular solution
Δ_A = Array(Δ)
KK = (κ) .* Δ_A
KK[1] = 1.0
s_A = Array(s)
pS = ifft(fft(s_A) ./ (-KK))
pS .+= mean(θ_A) 

Δ_A = Array(Δ)
colorrange = @lift((0, maximum($field)))
Kᵉ = effective_diffusivity[2] # 0.5 / maximum([sqrt(phase_speed), 1]) / 2 * amplitude_factor^2
field_diffusion = @lift(real.(ifft(fft(θ_A - pS * κ / Kᵉ) .* exp.(Δ_A * ($tindex - 0) * Kᵉ * Δt)) + pS * κ/ Kᵉ))
field_diffusion_slice = @lift($field_diffusion[:, floor(Int, N / 2)])

approximate_field = @lift(real.(ifft(fft(θ_A - pS) .* exp.(KK * ($tindex - 0) * Δt)) + pS))
approximate_field_slice = @lift($approximate_field[:, floor(Int, N / 2)])
heatmap!(ax11, x_A, x_A, field, colormap=colormap, interpolate=true, colorrange=colorrange)
heatmap!(ax21, x_A, x_A, field_diffusion, colormap=colormap, interpolate=true, colorrange=colorrange)
heatmap!(ax22, x_A, x_A, approximate_field, colormap=colormap, interpolate=true, colorrange=colorrange)
le = lines!(ax12, x_A, field_slice, color=:black)
ld = lines!(ax12, x_A, field_diffusion_slice, color=:red)
lnd = lines!(ax12, x_A, approximate_field_slice, color=:blue)
axislegend(ax12, [le, ld, lnd], ["ensemble", "effective diffusivity", "nonlocal diffusivity "], position=:rt)
display(fig)
end

##
#=
begin
fig = Figure(resolution=(1400, 1100))
ax11 = Axis(fig[1, 1]; title="ensemble average")
ax12 = Axis(fig[1, 2]; title="x=0 slice")
ax21 = Axis(fig[2, 1]; title="diffusion")
ax22 = Axis(fig[2, 2]; title="nonlocal space kernel")
t_slider = Slider(fig[3, 1:2], range=1:iend, startvalue=0)
tindex = t_slider.value
colormap = :bone_1
field = @lift(Array(θ̅_timeseries[:, :, $tindex]))
field_slice = @lift($field[:, floor(Int, N / 2)]) 

# particular solution
Δ_A = Array(Δ)
KK = (effective_diffusivity_operator .+ κ) .* Δ_A
KK[1] = 1.0
s_A = Array(s)
pS = ifft(fft(s_A) ./ (-KK))
pS .+= mean(θ_A) 

Δ_A = Array(Δ)
colorrange = @lift((0, maximum($field)))
Kᵉ = effective_diffusivity[2] # 0.5 / maximum([sqrt(phase_speed), 1]) / 2 * amplitude_factor^2
field_diffusion = @lift(real.(ifft(fft(θ_A - pS * κ / Kᵉ) .* exp.(Δ_A * ($tindex - 0) * Kᵉ * Δt)) + pS * κ/ Kᵉ))
field_diffusion_slice = @lift($field_diffusion[:, floor(Int, N / 2)])

approximate_field = @lift(real.(ifft(fft(θ_A - pS) .* exp.(KK * ($tindex - 0) * Δt)) + pS))
approximate_field_slice = @lift($approximate_field[:, floor(Int, N / 2)])
heatmap!(ax11, x_A, x_A, field, colormap=colormap, interpolate=true, colorrange=colorrange)
heatmap!(ax21, x_A, x_A, field_diffusion, colormap=colormap, interpolate=true, colorrange=colorrange)
heatmap!(ax22, x_A, x_A, approximate_field, colormap=colormap, interpolate=true, colorrange=colorrange)
le = lines!(ax12, x_A, field_slice, color=:black)
ld = lines!(ax12, x_A, field_diffusion_slice, color=:red)
lnd = lines!(ax12, x_A, approximate_field_slice, color=:blue)
axislegend(ax12, [le, ld, lnd], ["ensemble", "effective diffusivity", "nonlocal diffusivity "], position=:rt)
display(fig)
end
=#