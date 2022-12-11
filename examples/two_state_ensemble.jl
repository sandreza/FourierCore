using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2, GLMakie, HDF5
using ProgressBars
using CUDA
rng = MersenneTwister(1234)
Random.seed!(123456789)

# memo to self. Always draw twice for symmetry purposes

arraytype = CuArray
Ω = S¹(4π)^2
N = 2^7 # number of gridpoints

grid = FourierGrid(N, Ω, arraytype=arraytype)
nodes, wavenumbers = grid.nodes, grid.wavenumbers

x = nodes[1]
y = nodes[2]
kˣ = wavenumbers[1]
kʸ = wavenumbers[2]

##
# Fields 
# velocity
ψ = arraytype(zeros(ComplexF64, N, N))
u = similar(ψ)
v = similar(ψ)

# theta
θ = similar(ψ)
θ⁺= similar(ψ)
θ⁻= similar(ψ)
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
@. s = cos(kˣ[2] * x) * cos(kʸ[2] * y) # could also set source term to zero

# operators
∂x = im * kˣ
∂y = im * kʸ
Δ = @. ∂x^2 + ∂y^2

# plan ffts
P = plan_fft!(ψ)
P⁻¹ = plan_ifft!(ψ)

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

simulation_parameters = (; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, filter, ∂x, ∂y, κ, Δ, κΔθ)
size_of_A = size(A)

t = [0.0]
tend = 4*50.0 # 50.0 is good for the default
iend = ceil(Int, tend / Δt)
global Δt_old = Δt

realizations = 1000

rhs! = θ_rhs_symmetric!

# [10000.0, 25.0, 20.0, 15.0, 10.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.4, 0.3, 0.2, 0.1]
T = 5.0
nT = ceil(Int, T / Δt_old)
Δt = T / nT
iend = ceil(Int, tend / Δt)

θ̅_timeseries = CuArray(zeros(size(ψ)..., iend))
uθ_timeseries = CuArray(zeros(size(ψ)..., iend))
θ_timeseries = Array(zeros(size(ψ)..., iend))

ω = 2π / T
for j in ProgressBar(1:realizations)
    # new realization of flow
    rand!(rng, φ) # between 0, 1
    φ .*= 2π # to make it a random phase
    event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
    wait(event)

    t[1] = 0
    P * ψ # in place fft
    # ∇ᵖψ
    @. u = -1.0 * (∂y * ψ)
    # go back to real space 
    P⁻¹ * u
    P⁻¹ * ψ
    @. s = u * cos(ω * t[1]) + 0 * kʸ

    θ .= CuArray(θ_A)
    for i = 1:iend
        # fourth order runge kutta on deterministic part
        # keep ψ frozen is the correct way to do it here

        # the below assumes that φ is just a function of time
        rhs!(k₁, θ, simulation_parameters)
        @. θ̃ = θ + Δt * k₁ * 0.5
        t[1] += Δt / 2
        
        P * ψ # in place fft
        # ∇ᵖψ
        @. u = -1.0 * (∂y * ψ)
        # go back to real space 
        P⁻¹ * u
        P⁻¹ * ψ
        @. s = u * cos(ω * t[1]) + 0 * kʸ


        φ_rhs_normal!(φ̇, φ, rng)
        @. φ += phase_speed * sqrt(Δt / 2) * φ̇

        rhs!(k₂, θ̃, simulation_parameters)
        @. θ̃ = θ + Δt * k₂ * 0.5
        rhs!(k₃, θ̃, simulation_parameters)
        @. θ̃ = θ + Δt * k₃
        t[1] += Δt / 2
        
        P * ψ # in place fft
        # ∇ᵖψ
        @. u = -1.0 * (∂y * ψ)
        # go back to real space 
        P⁻¹ * u
        P⁻¹ * ψ
        @. s = u * cos(ω * t[1]) + 0 * kʸ


        φ_rhs_normal!(φ̇, φ, rng)
        @. φ += phase_speed * sqrt(Δt / 2) * φ̇

        rhs!(k₄, θ̃, simulation_parameters)
        @. θ += Δt / 6 * (k₁ + 2 * k₂ + 2 * k₃ + k₄)

        # update stochastic part 
        # φ_rhs_normal!(φ̇, φ, rng)
        # @. φ += sqrt(Δt) * φ̇

        # save output
        # tmp = real.(Array(θ))
        P⁻¹ * uθ
        @. θ̅_timeseries[:, :, i] += real.(θ) / realizations
        @. uθ_timeseries[:, :, i] += real.(uθ) / realizations
        if j == 1
            θ_timeseries[:, :, i] = Array(real.(θ))
        end

    end
    @. θ̅ += θ / realizations
end

toc = Base.time()
println("the time for the simulation was ", toc - tic, " seconds")

x_A = Array(x)[:] .- 2π
θ_F = Array(real.(θ))
θ̅_F = Array(real.(θ̅))

θ̅_timeseries_A = Array(θ̅_timeseries)
uθ_timeseries_A = Array(uθ_timeseries)
θ_timeseries_A = Array(θ_timeseries)

#=
begin
    start = 1 # - floor(Int, iend/2)
    skip = 10
    fig2 = Figure(resolution=(2*700, 2*300))
    ax11 = Axis(fig2[1, 1]; title="⟨θ⟩: averaged over y", xlabel="spatial index", ylabel="time index")
    ax21 = Axis(fig2[2, 1]; title="⟨θ⟩: black = index 32 of above, red = scaled forcing", xlabel="time index", ylabel="value")
    ax12 = Axis(fig2[1, 2]; title="⟨uθ⟩: averaged over y", xlabel="spatial index", ylabel="time index")
    ax22 = Axis(fig2[2, 2]; title="⟨uθ⟩: black = index 64 of above, red = scaled forcing", xlabel="time index", ylabel="value")

    mtheta2 = mean(θ̅_timeseries_A, dims=2)[:, 1, :]
    mutheta2 = mean(uθ_timeseries_A, dims=2)[:, 1, :]
    mtheta2max = maximum(mtheta2)
    mutheta2max = maximum(mutheta2)
    heatmap!(ax11, mtheta2[:, start:skip:iend], colorrange=(-mtheta2max, mtheta2max), colormap=:balance)
    heatmap!(ax12, mutheta2[:, start:skip:iend], colorrange=(-mutheta2max, mutheta2max), colormap=:balance)
    lines!(ax21, mtheta2[32, start:skip:iend], color=:black, linewidth=2)
    amp = maximum(mtheta2[32, start:skip:iend])
    lines!(ax21, amp .* cos.(ω .* collect(start:skip:iend) * Δt), color=:red, linewidth=2)

    lines!(ax22, mutheta2[64, start:skip:iend], color=:black, linewidth=2)
    amp = maximum(mutheta2[64, start:skip:iend])
    lines!(ax22, amp .* cos.(ω .* collect(start:skip:iend) * Δt), color=:red, linewidth=2)
    save("time_dependentSummary_plot_ω_" * string(ω)  * "_ensemble_" * string(realizations) * "_zeroth.png", fig2)
    using HDF5
    fid = h5open("time_dependent_ω_" * string(ω) * "_ensemble_" * string(realizations) * "_zeroth.hdf5", "w")
    fid["molecular_diffusivity"] = κ
    fid["streamfunction_amplitude"] = Array(A)
    fid["phase increase"] = phase_speed
    fid["time"] = collect(Δt * (1:iend))
    fid["omega"] = ω
    fid["ensemble mean"] = mtheta2
    fid["ensemble flux"] = mutheta2
    fid["ensemble number"] = realizations
    close(fid)
end
=#