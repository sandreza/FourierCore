using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2
rng = MersenneTwister(1234)
Random.seed!(123456789)

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
κ = 2 / N  # roughly 1/N for this flow
Δt = (x[2] - x[1]) / (4π)

# take the initial condition as negative of the source
# redo index 3
index_choices = 21:80
tic = Base.time()


for index_choice in index_choices

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

    for i = 1:iend
        event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
        wait(event)
        φ_rhs!(φ̇, φ, rng)
        θ_rhs!(θ̇, θ, params)
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

        if t[1] > 1000
            θ̅ .+= Δt * θ
        end

        if i % div(iend, 100) == 0
            println("time is t = ", t[1])
            local toc = Base.time()
            println("the time for the simulation is ", toc - tic, " seconds")
            println("extrema are ", extrema(real.(θ)))
            println("on wavenumber index ", index_choice)
        end

    end

    θ̅ ./= t[1]

    toc = Base.time()
    println("the time for the simulation was ", toc - tic, " seconds")
    println("saving ", "tracer_" * string(index_choice) * ".jld2")
    θ̅a = Array(real.(θ̅))
    jldsave("tracer_" * string(index_choice) * ".jld2"; ψ_save, θ_save, θ̅a)
end

#=
for i in eachindex(ψ_save)
    sleep(0.1)
    time_index[] = i
end
=#

