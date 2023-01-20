using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2
# GLMakie, 
using HDF5
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

@. θ = sin(kˣ[2] * x) * 6.4 * 0 + 0 * kʸ + 1.0 # 6.4 is roughly the ω = 0 case
θ_A = Array(θ)
θ̅ .= 0.0

ϵ = 1.0
simulation_parameters = (; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, filter, ∂x, ∂y, κ, Δ, κΔθ, ϵ)
size_of_A = size(A)

t = [0.0]
tend = 50.0 # 50.0 is good for the default
iend = ceil(Int, tend / Δt)
global Δt_old = Δt

realizations = 4000


rhs! = θ_rhs_compressible!

# T = 10.0
# for T in ProgressBar([10000.0, 25.0, 20.0, 15.0, 10.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.4, 0.3, 0.2, 0.1])
# nT = ceil(Int, T / Δt_old)
# Δt = T / nT
iend = ceil(Int, tend / Δt)

θ̅_timeseries = CuArray(zeros(size(ψ)..., iend))
uθ_timeseries = CuArray(zeros(size(ψ)..., iend))
θ_timeseries = Array(zeros(size(ψ)..., iend))

for j in ProgressBar(1:realizations)
    # new realization of flow
    rand!(rng, φ) # between 0, 1
    φ .*= 2π # to make it a random phase
    event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
    wait(event)

    t[1] = 0
    @. s = 0 # * sin(kˣ[2] * x) * cos(ω * t[1]) + 0 * kʸ

    θ .= CuArray(θ_A)
    for i = 1:iend
        # fourth order runge kutta on deterministic part
        # keep ψ frozen is the correct way to do it here

        # the below assumes that φ is just a function of time
        rhs!(k₁, θ, simulation_parameters)
        @. θ̃ = θ + Δt * k₁ * 0.5
        t[1] += Δt / 2
        # @. s = sin(kˣ[2] * x) * cos(ω * t[1]) + 0 * kʸ


        φ_rhs_normal!(φ̇, φ, rng)
        @. φ += phase_speed * sqrt(Δt / 2) * φ̇

        rhs!(k₂, θ̃, simulation_parameters)
        @. θ̃ = θ + Δt * k₂ * 0.5
        rhs!(k₃, θ̃, simulation_parameters)
        @. θ̃ = θ + Δt * k₃
        t[1] += Δt / 2
        # @. s = sin(kˣ[2] * x) * cos(ω * t[1]) + 0 * kʸ

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

θ̅_timeseries_A[:, :, end]
mean(uθ_timeseries_A[:, :, end])

    #=
    begin
        fig = Figure(resolution=(2048, 512))
        ax11 = Axis(fig[1, 1])
        ax12 = Axis(fig[2, 1])
        ax13 = Axis(fig[3, 1])
        t_slider = Slider(fig[4, 1:2], range=1:iend, startvalue=1)
        tindex = t_slider.value

        field1 = @lift(θ̅_timeseries_A[:, :, $tindex])
        field2 = @lift(θ_timeseries_A[:, :, $tindex])
        field3 = @lift(uθ_timeseries_A[:, :, $tindex])
        upbound = maximum(abs.(θ̅_timeseries_A[:, :, end]))
        hm1 = heatmap!(ax11, field1, colorrange=(-upbound, upbound), colormap=:balance)
        hm2 = heatmap!(ax12, field2, colorrange=(-upbound, upbound), colormap=:balance)
        hm3 = heatmap!(ax13, field3, colorrange=(-upbound, upbound), colormap=:balance)

        Colorbar(fig[1, 2], hm1, height=Relative(3 / 4), width=25, ticklabelsize=30, labelsize=30, ticksize=25, tickalign=1,)
        Colorbar(fig[2, 2], hm2, height=Relative(3 / 4), width=25, ticklabelsize=30, labelsize=30, ticksize=25, tickalign=1,)
        Colorbar(fig[3, 2], hm3, height=Relative(3 / 4), width=25, ticklabelsize=30, labelsize=30, ticksize=25, tickalign=1,)
        display(fig)
    end
    =#
    #=
    begin
        fig2 = Figure(resolution=(1400, 600))
        ax11 = Axis(fig2[1, 1]; title="⟨θ⟩: averaged over y", xlabel="spatial index", ylabel="time index")
        ax21 = Axis(fig2[2, 1]; title="⟨θ⟩: black = index 32 of above, red = scaled forcing", xlabel="time index", ylabel="value")
        ax12 = Axis(fig2[1, 2]; title="⟨uθ⟩: averaged over y", xlabel="spatial index", ylabel="time index")
        ax22 = Axis(fig2[2, 2]; title="⟨uθ⟩: black = index 64 of above, red = scaled forcing", xlabel="time index", ylabel="value")

        mtheta2 = mean(θ̅_timeseries_A, dims=2)[:, 1, :]
        mutheta2 = mean(uθ_timeseries_A, dims=2)[:, 1, :]
        mtheta2max = maximum(mtheta2)
        mutheta2max = maximum(mutheta2)
        heatmap!(ax11, mtheta2, colorrange=(-mtheta2max, mtheta2max), colormap=:balance)
        heatmap!(ax12, mutheta2, colorrange=(-mutheta2max, mutheta2max), colormap=:balance)
        lines!(ax21, mtheta2[32, :], color=:black, linewidth=2)
        amp = maximum(mtheta2[32, :])
        lines!(ax21, amp .* cos.(ω .* collect(1:iend) * Δt), color=:red, linewidth=2)

        lines!(ax22, mutheta2[64, :], color=:black, linewidth=2)
        amp = maximum(mutheta2[64, :])
        lines!(ax22, amp .* cos.(ω .* collect(1:iend) * Δt), color=:red, linewidth=2)
        save("time_dependentSummary_plot_ω_" * string(ω)  * "_ensemble_" * string(realizations) * ".png", fig2)
        using HDF5
        fid = h5open("compressible_ensemble_" * string(realizations) * ".hdf5", "w")
        fid["molecular_diffusivity"] = κ
        fid["streamfunction_amplitude"] = Array(A)
        fid["phase increase"] = phase_speed
        fid["time"] = collect(Δt * (1:iend))
        fid["epsilon"] = ϵ
        fid["ensemble mean"] = mtheta2
        fid["ensemble flux"] = mutheta2
        fid["ensemble number"] = realizations
        close(fid)
    end
end
=#