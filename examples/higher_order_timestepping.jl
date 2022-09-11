using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2
using ProgressBars
rng = MersenneTwister(1234)
Random.seed!(123456789)
# jld_name = "high_order_timestep_spatial_tracer_"
jld_name = "phase_speed_5_amplitude_factor_1"
include("transform.jl")
include("random_phase_kernel.jl")
# using GLMakie
save_fields = false
using CUDA
arraytype = CuArray
Ω = S¹(4π)^2
N = 2^7 # number of gridpoints
phase_speed = 5.0
amplitude_factor = 2.0

grid = FourierGrid(N, Ω, arraytype = arraytype)
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

##
# κ = 2 / N  # roughly 1/N for this flow
# κ = 2 / 2^8 # fixed diffusivity
# κ = 2e-4
κ = amplitude_factor * 2*Δx^2 
Δt = (x[2] - x[1]) / (4π) * 5 / amplitude_factor

# take the initial condition as negative of the source
index_choices = 2:80
tic = Base.time()

tstart = 1000
for index_choice in ProgressBar(index_choices)

    # save some snapshots
    ψ_save = typeof(real.(Array(ψ)))[]
    θ_save = typeof(real.(Array(ψ)))[]

    kᶠ = kˣ[index_choice]

    @. θ = cos(kᶠ * x) / (kᶠ)^2 / κ # scaling so that source is order 1
    θclims = extrema(Array(real.(θ))[:])
    P * θ # in place fft
    @. κΔθ = κ * Δ * θ
    P⁻¹ * κΔθ # in place fft
    s .= -κΔθ
    P⁻¹ * θ # in place fft
    θ̅ .= 0.0

    t = [0.0]
    tend = 5000 # 5000

    iend = ceil(Int, tend / Δt)

    params = (; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, s, P, P⁻¹, filter)

    size_of_A = size(A)

    for i = ProgressBar(1:iend)
        # fourth order runge kutta on deterministic part
        # keep ψ frozen is the correct way to do it here

        # the below assumes that φ is just a function of time
        θ_rhs_new!(k₁, θ, params)
        @. θ̃ = θ + Δt * k₁ * 0.5
    
        φ_rhs_normal!(φ̇, φ, rng)
        @. φ += phase_speed  * sqrt(Δt / 2) * φ̇
    
        θ_rhs_new!(k₂, θ̃, params)
        @. θ̃ = θ + Δt * k₂ * 0.5
        θ_rhs_new!(k₃, θ̃, params)
        @. θ̃ = θ + Δt * k₃

        φ_rhs_normal!(φ̇, φ, rng)
        @. φ += phase_speed  * sqrt(Δt / 2) * φ̇

        θ_rhs_new!(k₄, θ̃, params)
        @. θ += Δt / 6 * (k₁ + 2 * k₂ + 2 * k₃ + k₄)
    
        # update stochastic part 
        # φ_rhs_normal!(φ̇, φ, rng)
        # @. φ += sqrt(Δt) * φ̇
        
    
        t[1] += Δt
        # save output
    
        if save_fields
            if i % div(iend, 10) == 0
                # println("Saving at i=", i)
                push!(ψ_save, Array(real.(ψ)))
                push!(θ_save, Array(real.(θ)))
                # println("extrema are ", extrema(θ_save[end]))
                # println("time is t = ", t[1])
            end
        end
    
        if t[1] >= tstart
            θ̅ .+= Δt * θ
        end
        #=
        if i % div(iend, 100) == 0
            println("time is t = ", t[1])
            local toc = Base.time()
            println("the time for the simulation is ", toc - tic, " seconds")
            println("extrema are ", extrema(real.(θ)))
            println("on wavenumber index ", index_choice)
        end
        =#
    
    end

    θ̅ ./= (t[end] - tstart)

    toc = Base.time()
    # println("the time for the simulation was ", toc - tic, " seconds")
    # println("saving ", jld_name * string(index_choice) * ".jld2")
    θ̅a = Array(real.(θ̅))
    xnodes = Array(x)[:]
    ynodes = Array(y)[:]
    kˣ_wavenumbers = Array(kˣ)[:]
    kʸ_wavenumbers = Array(kˣ)[:]
    source = Array(s)
    if save_fields
        jldsave(jld_name * string(index_choice) * ".jld2"; ψ_save, θ_save, θ̅a, κ, xnodes, ynodes, kˣ_wavenumbers, kʸ_wavenumbers, source)
    else
        jldsave(jld_name * string(index_choice) * ".jld2"; θ̅a, κ, xnodes, ynodes, kˣ_wavenumbers, kʸ_wavenumbers, source)
    end
end

