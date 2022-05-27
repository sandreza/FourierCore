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
Ω = S¹(2π)^2
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
filter = @. abs(kˣ) .+ 0 * abs(kʸ) ≤ 2/3 * kxmax 
@. filter = filter * ( 0 * abs(kˣ) .+ 1 * abs(kʸ) ≤ 2/3 * kxmax )

# now define the random field 
wavemax = 5
𝓀 = arraytype([-wavemax, wavemax]) # arraytype(1.0 .* [-wavemax, -wavemax + 1, wavemax - 1, wavemax])# arraytype(collect(-wavemax:1:wavemax))
𝓀ˣ = reshape(𝓀, (length(𝓀), 1))
𝓀ʸ = reshape(𝓀, (1, length(𝓀)))
# A = @. 0.1 * (𝓀ˣ * 𝓀ˣ + 𝓀ʸ * 𝓀ʸ)^(-11 / 12)
A = @. 1.0 * (𝓀ˣ * 𝓀ˣ + 𝓀ʸ * 𝓀ʸ)^(0.0) # @. 1e-1 / (1 * 2 * wavemax^2) .* (𝓀ˣ * 𝓀ˣ + 𝓀ʸ * 𝓀ʸ)^(0.0) # ( 1 .+ (0 .* 𝓀ˣ) .* 𝓀ʸ) 
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
Δ⁻¹ = 1 ./ Δ
bools = (!).(isnan.(Δ⁻¹))
Δ⁻¹ .*= bools # hack in the fact that false * NaN = 0

# plan ffts
P = plan_fft!(ψ)
P⁻¹ = plan_ifft!(ψ)

# number of gridpoints in transition is about λ * N / 2
bump(x; λ=10 / N, width=π / 2) = 0.5 * (tanh((x + width / 2) / λ) - tanh((x - width / 2) / λ))
bumps(x; λ=20 / N, width=1.0) = 0.25 * (bump(x, λ=λ, width=width) + bump(x, λ=λ, width=2.0 * width) + bump(x, λ=λ, width=3.0 * width) + bump(x, λ=λ, width=4.0 * width))

##
Δx = x[2] - x[1]
κ = 1.0 / N * 0.1 # * 2^(2)  # roughly 1/N for this flow
# κ = 2 / 2^8 # fixed diffusivity
# κ = 2e-4
Δt = Δx / (2π) * 1

κ = 1.0 * Δx^2 # /Δt  # 1.0 / N * 0.1 # * 2^(2)  # roughly 1/N for this flow

# take the initial condition as negative of the source
tic = Base.time()

# save some snapshots
ψ_save = typeof(real.(Array(ψ)))[]
θ_save = typeof(real.(Array(ψ)))[]

r_A = Array(@. sqrt((x - π)^2 + (y - π)^2))
θ_A = [bumps(r_A[i, j]) - 3.0 for i in 1:N, j in 1:N]
θ .= CuArray(θ_A)
θ .= @. 0.1 * sin(3 * x) * sin(3 * y) + 0im
# @. θ = bump(sqrt(x^2 + y^2)) # scaling so that source is order 1
θclims = extrema(Array(real.(θ))[:])
P * θ # in place fft
@. κΔθ = κ * Δ * θ
P⁻¹ * κΔθ # in place fft
s .= -κΔθ * 0.0
P⁻¹ * θ # in place fft
θ̅ .= 0.0

t = [0.0]
tend = 200 # 5000

phase_speed = 1.0

iend = ceil(Int, tend / Δt)

#=
function θ_rhs_moisture!(θ̇, θ, params)
    #(; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, s, P, P⁻¹, filter) = params
    ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, s, P, P⁻¹, filter = params
    τ = 0.01
    e = 0.01
    event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
    wait(event)
    P * ψ # in place fft
    P * θ # in place fft
    # ∇ᵖψ
    @. u = filter * -1.0 * (∂y * ψ)
    @. v = filter * (∂x * ψ)
    # ∇θ
    @. ∂ˣθ = filter * ∂x * θ
    @. ∂ʸθ = filter * ∂y * θ
    @. κΔθ = κ * Δ * θ # (κ * Δ  - (κ * Δ)^2 ) * θ
    # go back to real space 
    P⁻¹ * ψ
    P⁻¹ * θ
    P⁻¹ * u
    P⁻¹ * v
    P⁻¹ * ∂ˣθ
    P⁻¹ * ∂ʸθ
    P⁻¹ * κΔθ
    # construct source 
    @. s = v + e - 1 / τ * θ * (real(θ) > 0)
    # Assemble RHS
    @. θ̇ = -u * ∂ˣθ - v * ∂ʸθ + κΔθ + s
    return nothing
end
=#


function θ_rhs_moisture!(θ̇, θ, params)
    #(; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, s, P, P⁻¹, filter) = params
    ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, s, P, P⁻¹, filter, Δ⁻¹ = params
    τ = 0.01
    e = 0.01
    event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
    wait(event)
    # @. ψ = cos(4 * x)
    @. s = ψ 
    P * ψ # in place fft
    P * θ # in place fft
    # quick hack 
    @. ψ = Δ⁻¹ * θ
    # ∇ᵖψ
    @. u = filter *  (∂y * ψ)
    @. v = filter * -1.0 * (∂x * ψ)
    # ∇θ
    @. ∂ˣθ = filter * ∂x * θ
    @. ∂ʸθ = filter * ∂y * θ
    @. κΔθ = (Δ⁻¹ + κ * Δ - 1e-1 * (κ * Δ)^2 + 1e-3 * (κ * Δ)^3 - 1e-5 * (κ * Δ)^4) * θ # κ * Δ - 0.1 * (κ * Δ)^2 + 0.01 * (κ * Δ)^3) * θ # κ * Δ * θ #
    # go back to real space 
    P⁻¹ * ψ
    P⁻¹ * θ
    P⁻¹ * u
    P⁻¹ * v
    P⁻¹ * ∂ˣθ
    P⁻¹ * ∂ʸθ
    P⁻¹ * κΔθ
    # construct source 
    # @. s = ψ# v + e - 1 / τ * θ * (real(θ) > 0)
    # Assemble RHS
    @. θ̇ = real(-u * ∂ˣθ - v * ∂ʸθ + κΔθ + s)
    @. θ = real(θ)
    #
    # P * θ̇    # in place fft
    # @. θ̇ = 0 * θ̇
    # P⁻¹ * θ̇
    return nothing
end


params = (; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, s, P, P⁻¹, filter, Δ⁻¹)

size_of_A = size(A)

realizations = 1
for j in 1:realizations
    # θ .= CuArray(θ_A)
    for i = 1:iend
        # fourth order runge kutta on deterministic part
        # keep ψ frozen is the correct way to do it here

        # the below assumes that φ is just a function of time
        θ_rhs_moisture!(k₁, θ, params)
        @. θ̃ = θ + Δt * k₁ * 0.5

        φ_rhs_normal!(φ̇, φ, rng)
        @. φ += phase_speed * sqrt(Δt / 2) * φ̇

        θ_rhs_moisture!(k₂, θ̃, params)
        @. θ̃ = θ + Δt * k₂ * 0.5
        θ_rhs_moisture!(k₃, θ̃, params)
        @. θ̃ = θ + Δt * k₃

        φ_rhs_normal!(φ̇, φ, rng)
        @. φ += phase_speed * sqrt(Δt / 2) * φ̇

        θ_rhs_moisture!(k₄, θ̃, params)
        @. θ += Δt / 6 * (k₁ + 2 * k₂ + 2 * k₃ + k₄)

        # update stochastic part 
        # φ_rhs_normal!(φ̇, φ, rng)
        # @. φ += sqrt(Δt) * φ̇


        t[1] += Δt
        # save output

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
        if i % div(iend, 100) == 0
            println("time is t = ", t[1])
            local toc = Base.time()
            println("the time for the simulation is ", toc - tic, " seconds")
            println("extrema are ", extrema(real.(θ)))
            # println("on wavenumber index ", index_choice)
        end

    end
    println("finished realization ", j)
    @. θ̅ += θ / realizations
end

toc = Base.time()
println("the time for the simulation was ", toc - tic, " seconds")

# calculate vorticity
P * ψ # in place fft
@. κΔθ = Δ * ψ
P⁻¹ * κΔθ # in place fft

x_A = Array(x)[:] .- 2π
θ_F = Array(real.(θ))
ψ_F = Array(real.(ψ))
ω_F = Array(real.(κΔθ))
θ̅_F = Array(real.(θ̅))

fig = Figure(resolution=(2048, 512))
ax1 = Axis(fig[1, 1], title="moisture t = 0")
ax2 = Axis(fig[1, 2], title="moisture t = " * string(tend))
ax3 = Axis(fig[1, 4], title="vorticity t = " * string(tend))
println("the extrema of the end field is ", extrema(θ_F))
println("the extrema of the ensemble average is ", extrema(θ̅_F))
colormap = :bone_1
# colormap = :nipy_spectral
colorrange = (-1, 0)
colorrange2 = extrema(θ_F)
colorrange3 = (-maximum(ω_F), maximum(ω_F))
heatmap!(ax1, x_A, x_A, θ_A, colormap=colormap, colorrange=colorrange2, interpolate=true)
hm = heatmap!(ax2, x_A, x_A, θ_F, colormap=colormap, colorrange=colorrange2, interpolate=true)
hm_e = heatmap!(ax3, x_A, x_A, ω_F, colormap=:balance, colorrange=colorrange3, interpolate=true)
Colorbar(fig[1, 3], hm, height=Relative(3 / 4), width=25, ticklabelsize=30, labelsize=30, ticksize=25, tickalign=1,)
Colorbar(fig[1, 5], hm_e, height=Relative(3 / 4), width=25, ticklabelsize=30, labelsize=30, ticksize=25, tickalign=1,)
display(fig)
