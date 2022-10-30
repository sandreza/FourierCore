using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2, GLMakie, HDF5
using ProgressBars
rng = MersenneTwister(1234)
Random.seed!(123456789)
# jld_name = "high_order_timestep_spatial_tracer_"
jld_name = "blocky_"
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
amplitude_factor = 1.0
phase_speed = 1.0

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

# number of gridpoints in transition is about λ * N / 2
bump(x; λ=20 / N, width=π / 2) = 0.5 * (tanh((x + width / 2) / λ) - tanh((x - width / 2) / λ))
bumps(x; λ=20 / N, width=1.0) = 0.25 * (bump(x, λ=λ, width=width) + bump(x, λ=λ, width=2.0 * width) + bump(x, λ=λ, width=3.0 * width) + bump(x, λ=λ, width=4.0 * width))

##
Δx = x[2] - x[1]
κ = 0.01 # * (2^7 / N)^2# amplitude_factor * 2 * Δx^2
cfl = 0.1
Δx = (x[2] - x[1])
advective_Δt = cfl * Δx / amplitude_factor
diffusive_Δt = cfl * Δx^2 / κ
Δt = minimum([advective_Δt, diffusive_Δt])

# take the initial condition as negative of the source
tic = Base.time()

# save some snapshots
ψ_save = typeof(real.(Array(ψ)))[]
θ_save = typeof(real.(Array(ψ)))[]

r_A = Array(@. sqrt((x - 2π)^2 + (y - 2π)^2))

@. θ = sin(kˣ[2] * x) * 6.4 * 0 + 0 * kʸ # 6.4 is roughly the ω = 0 case
θ_A = Array(θ)
θ̅ .= 0.0

t = [0.0]
tend = 100.0 # 5 

iend = ceil(Int, tend / Δt)

simulation_parameters = (; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, filter)

size_of_A = size(A)

realizations = 100

θ̅_timeseries = CuArray(zeros(size(ψ)..., iend))
θ_timeseries = Array(zeros(size(ψ)..., iend))
rhs! = θ_rhs_symmetric!


ω = 2π / 1.0
for j in ProgressBar(1:realizations)
    # new realization of flow
    rand!(rng, φ) # between 0, 1
    φ .*= 2π # to make it a random phase
    event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
    wait(event)

    t[1] = 0
    @. s = sin(kˣ[2] * x )* cos(ω * t[1]) + 0 * kʸ

    θ .= CuArray(θ_A)
    for i = 1:iend
        # fourth order runge kutta on deterministic part
        # keep ψ frozen is the correct way to do it here

        # the below assumes that φ is just a function of time
        rhs!(k₁, θ, simulation_parameters)
        @. θ̃ = θ + Δt * k₁ * 0.5
        t[1] += Δt / 2
        @. s = sin(kˣ[2] * x) * cos(ω * t[1]) + 0 * kʸ


        φ_rhs_normal!(φ̇, φ, rng)
        @. φ += phase_speed * sqrt(Δt / 2) * φ̇

        rhs!(k₂, θ̃, simulation_parameters)
        @. θ̃ = θ + Δt * k₂ * 0.5
        rhs!(k₃, θ̃, simulation_parameters)
        @. θ̃ = θ + Δt * k₃
        t[1] += Δt / 2
        @. s = sin(kˣ[2] * x) * cos(ω * t[1]) + 0 * kʸ

        φ_rhs_normal!(φ̇, φ, rng)
        @. φ += phase_speed * sqrt(Δt / 2) * φ̇

        rhs!(k₄, θ̃, simulation_parameters)
        @. θ += Δt / 6 * (k₁ + 2 * k₂ + 2 * k₃ + k₄)

        # update stochastic part 
        # φ_rhs_normal!(φ̇, φ, rng)
        # @. φ += sqrt(Δt) * φ̇

        # save output
        # tmp = real.(Array(θ))
        @. θ̅_timeseries[:, :, i] += real.(θ) / realizations
        if j == 1
            θ_timeseries[:, :, i] = Array(real.(θ))
        end
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

θ̅_timeseries_A = Array(θ̅_timeseries)
θ_timeseries_A = Array(θ_timeseries)

begin
    fig = Figure(resolution=(2048, 512))
    ax11 = Axis(fig[1, 1])
    ax12 = Axis(fig[2, 1])
    t_slider = Slider(fig[3, 1:2], range=1:iend, startvalue=1)
    tindex = t_slider.value

    field1 = @lift(θ̅_timeseries_A[:, :, $tindex])
    field2 = @lift(θ_timeseries_A[:, :, $tindex])
    hm1 = heatmap!(ax11, field1, colorrange=(-0.35, 0.35), colormap=:balance)
    hm2 = heatmap!(ax12, field2, colorrange=(-0.35, 0.35), colormap=:balance)

    Colorbar(fig[1, 2], hm1, height=Relative(3 / 4), width=25, ticklabelsize=30, labelsize=30, ticksize=25, tickalign=1,)
    Colorbar(fig[2, 2], hm2, height=Relative(3 / 4), width=25, ticklabelsize=30, labelsize=30, ticksize=25, tickalign=1,)
    display(fig)
end

fig2 = Figure(resolution=(2048, 512))
ax11 = Axis(fig2[1, 1])
ax12 = Axis(fig2[2, 1])
mtheta2 = mean(θ̅_timeseries_A, dims=2)[:, 1, :]
heatmap!(ax11, mtheta2, colorrange=(-0.35, 0.35), colormap=:balance)
lines!(ax12, mtheta2[10, :], color=:black, linewidth=2)
amp = maximum(mtheta2[10, :])
lines!(ax12, amp .* sin.(ω .* collect(1:iend) * Δt), color=:red, linewidth=2)

##
#=
begin
    fig = Figure(resolution=(1400, 1100))
    t_slider = Slider(fig[3, 1:2], range=1:iend, startvalue=0)
    tindex = t_slider.value

    ax11 = Axis(fig[1, 1]; title="ensemble average")
    ax12 = Axis(fig[1, 2]; title=@lift("x=0 slice at t = " * string($tindex * Δt)))
    ax21 = Axis(fig[2, 1]; title="diffusion")
    ax22 = Axis(fig[2, 2]; title="nonlocal space kernel")

    colormap = :bone_1
    field = @lift(Array(θ̅_timeseries[:, :, $tindex]))
    field_slice = @lift($field[:, floor(Int, N / 2)])

    # particular solution
    Δ_A = Array(Δ)
    KK = (κ) .* Δ_A
    KK[1] = 1.0
    s_A = Array(s)
    pS = ifft(fft(s_A) ./ (-KK))
    KK[1] = 0.0
    pS = pS .- mean(pS) .+ mean(θ_A)

    Δ_A = Array(Δ)
    colorrange = @lift((0, maximum($field)))
    Kᵉ = amplitude_factor^2 # effective_diffusivity[2] # 0.5 / maximum([sqrt(phase_speed), 1]) / 2 * amplitude_factor^2
    field_diffusion = @lift(real.(ifft(fft(θ_A - pS * κ / Kᵉ) .* exp.(Δ_A * ($tindex - 0) * Kᵉ * Δt)) + pS * κ / Kᵉ))
    field_diffusion_slice = @lift($field_diffusion[:, floor(Int, N / 2)])

    approximate_field = @lift(real.(ifft(fft(θ_A - pS) .* exp.(KK * ($tindex - 0) * Δt)) + pS))
    approximate_field_slice = @lift($approximate_field[:, floor(Int, N / 2)])
    heatmap!(ax11, x_A, x_A, field, colormap=colormap, interpolate=true, colorrange=colorrange)
    heatmap!(ax21, x_A, x_A, field_diffusion, colormap=colormap, interpolate=true, colorrange=colorrange)
    heatmap!(ax22, x_A, x_A, approximate_field, colormap=colormap, interpolate=true, colorrange=colorrange)
    le = lines!(ax12, x_A, field_slice, color=:black)
    ld = lines!(ax12, x_A, field_diffusion_slice, color=:red)
    lnd = lines!(ax12, x_A, approximate_field_slice, color=:blue)
    axislegend(ax12, [le, ld, lnd], ["ensemble", "effective diffusivity", "molecular diffusivity "], position=:rt)
    display(fig)
end
=#