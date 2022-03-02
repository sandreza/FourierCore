using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2, Statistics
rng = MersenneTwister(1234)
Random.seed!(123456789)
jld_name = "high_res_tracer_"
include("transform.jl")
include("random_phase_kernel.jl")
# using GLMakie
using CUDA
arraytype = CuArray
Ω = S¹(4π)^2
N = 2^8 # number of gridpoints
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
A = @. 0.1 * (𝓀ˣ * 𝓀ˣ + 𝓀ʸ * 𝓀ʸ)^(-11 / 12)
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
κ = 2 / 2^8  # roughly 1/N for this flow
Δt = (x[2] - x[1]) / (4π)

# take the initial condition as negative of the source
# redo index 3
index_choices = [1]
tic = Base.time()

tstart = 1000
for index_choice in index_choices

    # save some snapshots
    ψ_save = typeof(real.(Array(ψ)))[]
    θ_save = typeof(real.(Array(ψ)))[]

    kᶠ = kˣ[2]

    @. θ = cos(kᶠ * x) / (kᶠ)^2 / κ # scaling so that source is order 1
    θclims = extrema(Array(real.(θ))[:])
    P * θ # in place fft
    @. κΔθ = κ * Δ * θ
    P⁻¹ * κΔθ # in place fft
    s .= -κΔθ
    P⁻¹ * θ # in place fft

    # construct initial condition for θ
    event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
    wait(event)
    φ_rhs!(φ̇, φ, rng)
    P * ψ # in place fft
    @. u = -1.0 * (∂y * ψ)
    P⁻¹ * u
    P⁻¹ * ψ
    θ .= u

    θ̅ .= 0.0

    t = [0.0]
    tend = 5000 # 5000

    iend = ceil(Int, tend / Δt)

    params = (; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, s, P, P⁻¹, filter)

    size_of_A = size(A)

    for i = 1:iend
        event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
        wait(event)
        φ_rhs!(φ̇, φ, rng)
        θ_rhs_zeroth!(θ̇, θ, params)
        # Euler step
        @. φ += sqrt(Δt) * φ̇
        @. θ += Δt * θ̇
        t[1] += Δt
        # save output

        if i % div(iend, 10) == 0
            println("Saving at i=", i)
            push!(ψ_save, Array(real.(ψ)))
            push!(θ_save, Array(real.(θ)))
            println("extrema are ", extrema(θ_save[end]))
            println("time is t = ", t[1])
        end

        if t[1] >= tstart
            P * ψ # in place fft
            # ∇ᵖψ
            @. u = -1.0 * (∂y * ψ)
            P⁻¹ * u
            P⁻¹ * ψ # in place fft

            @. θ̅ += Δt * u * θ # direct accumation of fluxes, admittedly mislabed but w/e
        end

        if i % div(iend, 100) == 0
            println("time is t = ", t[1])
            local toc = Base.time()
            println("the time for the simulation is ", toc - tic, " seconds")
            println("extrema are ", extrema(real.(θ)))
            println(" the current ⟨θ̅⟩ is ", mean(θ̅))
            println("on wavenumber index ", index_choice)
        end

    end

    θ̅ ./= (t[end] - tstart)

    toc = Base.time()
    println("the time for the simulation was ", toc - tic, " seconds")
    println("saving ", jld_name * string(index_choice) * ".jld2")
    θ̅a = Array(real.(θ̅))
    xnodes = Array(x)[:]
    ynodes = Array(y)[:]
    kˣ_wavenumbers = Array(kˣ)[:]
    kʸ_wavenumbers = Array(kˣ)[:]
    jldsave(jld_name * string(index_choice) * ".jld2"; ψ_save, θ_save, θ̅a, κ, xnodes, ynodes, kˣ_wavenumbers, kʸ_wavenumbers)
end
