using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, HDF5, ProgressBars, Statistics
using CUDA
arraytype = CuArray

rng = MersenneTwister(12345)
Random.seed!(12)
initialize = false
tend = 200.0
tstart = 1
phase_speed = sqrt(1.0) # 1.0

N = 2^7
N_ens = 2^7
Ns = (N, N, N_ens)

κ = 1e-3 # 1.0 * Δx^2
ν = sqrt(1e-5) # 0.5 * Δx^2
ν_h = sqrt(1e-3) # 0.001
f_amp = 400
ϵ = 1.0

function load_psi!(ψ)
    filename = "initial_streamfunction.hdf5"
    fid = h5open(filename, "r")
    ψ .= arraytype(read(fid["psi"]))
    close(fid)
    return nothing
end



Ω = S¹(4π)^2 × S¹(1)
grid = FourierGrid(Ns, Ω, arraytype=arraytype)
nodes, wavenumbers = grid.nodes, grid.wavenumbers
forcing_amplitude = f_amp * (N / 2^7)^2 # due to FFT nonsense

# build filter 
x = nodes[1]
y = nodes[2]
kˣ = wavenumbers[1]
kʸ = wavenumbers[2]
# construct waver
kxmax = maximum(kˣ)
kymax = maximum(kˣ)
kxmax = kymax = 30
waver = @. (kˣ)^2 + (kʸ)^2 ≤ ((kxmax / 2)^2 + (kymax / 2)^2)
waver = @. abs(kˣ) .+ 0 * abs(kʸ) ≤ 2 / 3 * kxmax
@. waver = waver * (0 * abs(kˣ) .+ 1 * abs(kʸ) ≤ 2 / 3 * kxmax)
waver[1, 1] = 0.0
waver[:, floor(Int, N / 2)+1] .= 0.0
waver[floor(Int, N / 2)+1, :] .= 0.0

# construct fields 
φ = arraytype(zeros(Ns...))
rand!(rng, φ)
φ *= 2π

field = arraytype(zeros(N, N))

##
# Fields 
# velocity
ψ = arraytype(zeros(ComplexF64, Ns...))
u = similar(ψ)
v = similar(ψ)
u₀ = similar(ψ)
v₀ = similar(ψ)

# auxiliary fields
S = arraytype(zeros(ComplexF64, size(ψ)..., 2))
Ṡ = arraytype(zeros(ComplexF64, size(ψ)..., 2))

uθ = similar(ψ)
vθ = similar(ψ)
uζ = similar(ψ)
vζ = similar(ψ)


∂ˣθ = similar(ψ)
∂ʸθ = similar(ψ)
∂ˣζ = similar(ψ)
∂ʸζ = similar(ψ)

∂ˣuθ = similar(ψ)
∂ʸvθ = similar(ψ)
∂ˣuζ = similar(ψ)
∂ʸvζ = similar(ψ)

𝒟θ = similar(ψ)
𝒟ζ = similar(ψ)
θ̇ = similar(ψ)

k₁ = similar(S)
k₂ = similar(S)
k₃ = similar(S)
k₄ = similar(S)
S̃ = similar(S)

# source
sθ = similar(ψ)
sζ = similar(ψ)

# phase
φ̇ = similar(φ)

# operators
∂x = im * kˣ
∂y = im * kʸ
Δ = @. ∂x^2 + ∂y^2

# plan ffts
P = plan_fft!(ψ, (1, 2))
P⁻¹ = plan_ifft!(ψ, (1, 2))

# time stepping
Δx = x[2] - x[1]
Δt = 1 / N # 1 / N # 2^(-10)# 2 / N # Δx / (2π) * 1.0
r = 0
hypoviscocity_power = 2
dissipation_power = 2
# Dissipation 
Δ = @. ∂x^2 + ∂y^2
Δ⁻¹ = 1 ./ Δ
bools = (!).(isnan.(Δ⁻¹))
Δ⁻¹ .*= bools # hack in the fact that false * NaN = 0

𝒟ν = @. -(-ν_h * Δ⁻¹)^(hypoviscocity_power) - (-ν * Δ)^(dissipation_power) - r # - 1e-1 * (κ * Δ)^2 + 1e-3 * (κ * Δ)^3 - 1e-5 * (κ * Δ)^4
𝒟κ = @. κ * Δ # - 1e-1 * (κ * Δ)^2 + 1e-3 * (κ * Δ)^3 - 1e-5 * (κ * Δ)^4

##
θ = view(S, :, :, :, 1)
ζ = view(S, :, :, :, 2)
@. θ = 0.0 * sin(3 * x) * sin(3 * y) + -0.1 + 0im
@. ζ = sin(3 * x) * sin(3 * y)

operators = (; P, P⁻¹, Δ⁻¹, waver, 𝒟ν, 𝒟κ, ∂x, ∂y)
auxiliary = (; ψ, x, y, φ, u, v, uζ, vζ, uθ, vθ, ∂ˣζ, ∂ʸζ, ∂ˣθ, ∂ʸθ, ∂ˣuζ, ∂ʸvζ, ∂ˣuθ, ∂ʸvθ, 𝒟θ, 𝒟ζ, sθ, sζ)
constants = (; forcing_amplitude=forcing_amplitude, ϵ = ϵ)# (; τ = 0.01, e = 0.01)
parameters = (; auxiliary, operators, constants)

# initialize
if initialize != true
    load_psi!(ψ)
    ζ .= ifft(Δ .* fft(ψ))
end
sθ .= 0.0

##
function rhs!(Ṡ, S, parameters)
    θ̇ = view(Ṡ, :, :, :, 1)
    ζ̇ = view(Ṡ, :, :, :, 2)
    θ = view(S, :, :, :, 1)
    ζ = view(S, :, :, :, 2)

    (; P, P⁻¹, Δ⁻¹, waver, 𝒟ν, 𝒟κ, ∂x, ∂y) = parameters.operators
    (; ψ, x, y, φ, u, v, uζ, vζ, uθ, vθ, ∂ˣζ, ∂ʸζ, ∂ˣθ, ∂ʸθ, ∂ˣuζ, ∂ʸvζ, ∂ˣuθ, ∂ʸvθ, 𝒟θ, 𝒟ζ, sθ, sζ) = parameters.auxiliary
    (; forcing_amplitude, ϵ) = parameters.constants

    # construct source for vorticity 
    # @. sζ = ψ
    sζ .= waver .* forcing_amplitude .* exp.(im .* φ)
    P⁻¹ * sζ

    P * θ # in place fft ζ
    P * ζ # in place fft
    # grab stream function from vorticity
    @. ψ = Δ⁻¹ * ζ
    # ∇ᵖψ
    @. u = (∂y * ψ)
    @. v = -1.0 * (∂x * ψ)
    # ∇ζ
    @. ∂ˣθ = ∂x * θ
    @. ∂ʸθ = ∂y * θ
    @. ∂ˣζ = ∂x * ζ
    @. ∂ʸζ = ∂y * ζ
    # Dissipation
    @. 𝒟ζ = 𝒟ν * ζ
    @. 𝒟θ = 𝒟κ * θ
    # go back to real space 
    P⁻¹ * u
    P⁻¹ * v
    P⁻¹ * ζ
    P⁻¹ * ∂ˣζ
    P⁻¹ * ∂ʸζ
    P⁻¹ * 𝒟ζ
    P⁻¹ * θ
    P⁻¹ * ∂ˣθ
    P⁻¹ * ∂ʸθ
    P⁻¹ * 𝒟θ
    # construct conservative form 
    @. uζ = u * ζ
    @. vζ = v * ζ
    @. uθ = u * θ
    @. vθ = v * θ
    # in place fft 
    P * uζ
    P * vζ
    P * uθ
    P * vθ
    # ∇⋅(u⃗ζ)
    @. ∂ˣuζ = ∂x * uζ
    @. ∂ʸvζ = ∂y * vζ
    # ∇⋅(u⃗θ)
    @. ∂ˣuθ = ∂x * uθ
    @. ∂ʸvθ = ∂y * vθ
    # in place ifft 
    P⁻¹ * ∂ˣuζ
    P⁻¹ * ∂ʸvζ
    P⁻¹ * ∂ˣuθ
    P⁻¹ * ∂ʸvθ

    # rhs
    @. ζ̇ = real((-u * ∂ˣζ - v * ∂ʸζ - ∂ˣuζ - ∂ʸvζ) * 0.5 + 𝒟ζ + sζ)
    @. θ̇ = real((-u * ∂ˣθ - v * ∂ʸθ - ∂ˣuθ - ∂ʸvθ) * 0.5 * ϵ + 𝒟θ + sθ)
    @. S = real(S)
    @. Ṡ = real(Ṡ)

    return nothing
end

function step!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, parameters)
    rhs!(k₁, S, parameters)
    @. S̃ = S + Δt * k₁ * 0.5
    randn!(rng, φ̇)
    @. φ += phase_speed * sqrt(Δt / 2) * φ̇ # now at t = 0.5
    rhs!(k₂, S̃, parameters)
    @. S̃ = S + Δt * k₂ * 0.5
    rhs!(k₃, S̃, parameters)
    @. S̃ = S + Δt * k₃
    randn!(rng, φ̇)
    @. φ += phase_speed * sqrt(Δt / 2) * φ̇ # now at t = 1.0
    rhs!(k₄, S̃, parameters)
    @. S += Δt / 6 * (k₁ + 2 * k₂ + 2 * k₃ + k₄)
    return nothing
end
##
iend = ceil(Int, tend / Δt)
start_index = floor(Int, tstart / Δt)
eulerian_list = Float64[]
lagrangian_list = Float64[]
eke_list = Float64[]

iter = ProgressBar(1:iend)
if initialize
    for i = iter
        # fourth order runge-kutta on deterministic part
        # keep ψ frozen is the correct way to do it here

        # the below assumes that φ is just a function of time
        rhs!(k₁, S, parameters)
        @. S̃ = S + Δt * k₁ * 0.5

        # φ_rhs_normal!(φ̇, φ, rng)
        randn!(rng, φ̇)
        @. φ += phase_speed * sqrt(Δt / 2) * φ̇ # now at t = 0.5

        rhs!(k₂, S̃, parameters)
        @. S̃ = S + Δt * k₂ * 0.5
        rhs!(k₃, S̃, parameters)
        @. S̃ = S + Δt * k₃

        # φ_rhs_normal!(φ̇, φ, rng)
        randn!(rng, φ̇)
        @. φ += phase_speed * sqrt(Δt / 2) * φ̇ # now at t = 1.0

        rhs!(k₄, S̃, parameters)
        @. S += Δt / 6 * (k₁ + 2 * k₂ + 2 * k₃ + k₄)

        t[1] += Δt

        if i == start_index
            θ .= u
            u₀ .= u
        end
        if (i > start_index) && (i % 10 == 0)
            if i % 4000 == 0
                # decorrelates after 2000 timesteps
                θ .= u
            end
            uu = real(mean(u .* u₀))
            uθ = real(mean(u .* θ))

            push!(eulerian_list, uu)
            push!(lagrangian_list, uθ)
            push!(eke_list, real(0.5 * mean(u .* u + v .* v)))

            θ_min, θ_max = extrema(real.(θ))
            ζ_min, ζ_max = extrema(real.(ζ))
            set_multiline_postfix(iter, "θ_min: $θ_min \nθ_max: $θ_max \nζ_min: $ζ_min \nζ_max: $ζ_max")
        end


    end
end

if initialize
    filename = "initial_streamfunction.hdf5"
    fid = h5open(filename, "w")
    P⁻¹ * ψ
    fid["psi"] = Array(ψ)
    close(fid)
end

#=
if initialize
    using GLMakie
    fig2 = Figure()

    ax = Axis(fig2[1, 1])
    A_ζ = Array(real.(ζ))
    tmp = quantile(abs.(extrema(A_ζ))[:], 0.1)
    ax2 = Axis(fig2[1, 2])
    A_θ = Array(real.(θ))
    tmp2 = quantile(abs.(extrema(A_θ))[:], 0.1)

    sl_x = Slider(fig2[2, 1:2], range=1:N_ens, startvalue=1)
    o_index = sl_x.value

    field = @lift Array(ζ[:, :, $o_index])
    heatmap!(ax, field, colormap=:balance, colorrange=(-tmp, tmp), interpolate=false)

    field2 = @lift Array(θ[:, :, $o_index])
    heatmap!(ax2, field2, colormap=:balance, colorrange=(-tmp2, tmp2), interpolate=false)
    display(fig2)
end
=#