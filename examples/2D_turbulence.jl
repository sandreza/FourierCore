using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, HDF5, ProgressBars
using GLMakie

# NOTE ALIASING BUG FIXED ∂x[N/2] = 0

rng = MersenneTwister(12345)
# Random.seed!(123456789)
Random.seed!(12)

# jld_name = "high_order_timestep_spatial_tracer_"
# jld_name = "blocky"
# include("transform.jl")
include("random_phase_kernel.jl")
# using GLMakie

using CUDA
arraytype = CuArray
Ω = S¹(2π)^2
N = 2^7
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
filter = @. abs(kˣ) .+ 0 * abs(kʸ) ≤ 2 / 3 * kxmax
@. filter = filter * (0 * abs(kˣ) .+ 1 * abs(kʸ) ≤ 2 / 3 * kxmax)

# DEFINE TIME END 
tend = 500
# now define the random field 
wavemax = 7
𝓀 = arraytype([-wavemax, 0.0, wavemax]) # arraytype(1.0 .* [-wavemax, -wavemax + 1, wavemax - 1, wavemax])# arraytype(collect(-wavemax:1:wavemax))
𝓀ˣ = reshape(𝓀, (length(𝓀), 1))
𝓀ʸ = reshape(𝓀, (1, length(𝓀)))
# A = @. 0.1 * (𝓀ˣ * 𝓀ˣ + 𝓀ʸ * 𝓀ʸ)^(-11 / 12)
A = @. 0.5 * (𝓀ˣ * 𝓀ˣ + 𝓀ʸ * 𝓀ʸ) * (0.00) + 1 / 8 # 10 * 0.5 # @. 1e-1 / (1 * 2 * wavemax^2) .* (𝓀ˣ * 𝓀ˣ + 𝓀ʸ * 𝓀ʸ)^(0.0) # ( 1 .+ (0 .* 𝓀ˣ) .* 𝓀ʸ) 
A[A.==Inf] .= 0.0
A[2, 2] = 0.0 # so that vorticity is mean zero
φ = arraytype(2π * rand(size(A)...))
field = arraytype(zeros(N, N))

##
# Fields 
# velocity
ψ = arraytype(zeros(ComplexF64, N, N))
u = similar(ψ)
v = similar(ψ)

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

##
Δx = x[2] - x[1]
Δt = Δx / (2π) * 1.0
κ = 1.0 * Δx^2
ν = 0.5 * Δx^2
ν_h = 0.001
hypoviscocity_power = 1
dissipation_power = 2
# Dissipation 
𝒟ν = @. -(-ν_h * Δ⁻¹)^(hypoviscocity_power) - (-ν * Δ)^(dissipation_power) # - 1e-1 * (κ * Δ)^2 + 1e-3 * (κ * Δ)^3 - 1e-5 * (κ * Δ)^4
𝒟κ = @. κ * Δ # - 1e-1 * (κ * Δ)^2 + 1e-3 * (κ * Δ)^3 - 1e-5 * (κ * Δ)^4

# take the initial condition as negative of the source
tic = Base.time()

# save some snapshots
ψ_save = typeof(real.(Array(ψ)))[]
θ_save = typeof(real.(Array(ψ)))[]

θ = view(S, :, :, 1)
ζ = view(S, :, :, 2)
@. θ = 0.0 * sin(3 * x) * sin(3 * y) + -0.1 + 0im
@. ζ = sin(3 * x) * sin(3 * y)

t = [0.0]

phase_speed = sqrt(1.0) # 1.0

iend = ceil(Int, tend / Δt)

operators = (; P, P⁻¹, Δ⁻¹, filter, 𝒟ν, 𝒟κ, ∂x, ∂y)
auxiliary = (; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, uζ, vζ, uθ, vθ, ∂ˣζ, ∂ʸζ, ∂ˣθ, ∂ʸθ, ∂ˣuζ, ∂ʸvζ, ∂ˣuθ, ∂ʸvθ, 𝒟θ, 𝒟ζ, sθ, sζ)
constants = (; τ=4.0 * Δt, e=1e-2)# (; τ = 0.01, e = 0.01)

parameters = (; auxiliary, operators, constants)

function rhs!(Ṡ, S, parameters)
    θ̇ = view(Ṡ, :, :, 1)
    ζ̇ = view(Ṡ, :, :, 2)
    θ = view(S, :, :, 1)
    ζ = view(S, :, :, 2)


    (; P, P⁻¹, Δ⁻¹, filter, 𝒟ν, 𝒟κ, ∂x, ∂y) = parameters.operators
    (; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, uζ, vζ, uθ, vθ, ∂ˣζ, ∂ʸζ, ∂ˣθ, ∂ʸθ, ∂ˣuζ, ∂ʸvζ, ∂ˣuθ, ∂ʸvθ, 𝒟θ, 𝒟ζ, sθ, sζ) = parameters.auxiliary
    τ, e = parameters.constants


    # construct random phase forcing
    event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
    wait(event)

    # construct source for vorticity
    @. sζ = ψ

    # P * ψ
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

    # construct source 
    # @. sθ = v + e - 1 / τ * θ * (real(θ) > 0)

    # rhs
    @. ζ̇ = real((-u * ∂ˣζ - v * ∂ʸζ - ∂ˣuζ - ∂ʸvζ) * 0.5 + 𝒟ζ + sζ)
    @. θ̇ = real((-u * ∂ˣθ - v * ∂ʸθ - ∂ˣuθ - ∂ʸvθ) * 0.5 + 𝒟θ)
    @. S = real(S)
    @. Ṡ = real(Ṡ)

    return nothing
end

include("interpolation.jl")
filename = "higher_rez.hdf5"
# rm(filename)
# fid = h5open(filename, "w")
# create_group(fid, "vorticity")
# create_group(fid, "moisture")
saveindex = 0

start_index = floor(Int, iend / 100)
θ_t = [] # zeros(N, N, iend-start_index)
u_t = [] # zeros(N, N, iend-start_index)
v_t = [] # zeros(N, N, iend-start_index)
ζ_t = []

iter = ProgressBar(1:iend)
for i = iter
    # fourth order runge-kutta on deterministic part
    # keep ψ frozen is the correct way to do it here

    # the below assumes that φ is just a function of time
    rhs!(k₁, S, parameters)
    @. S̃ = S + Δt * k₁ * 0.5

    φ_rhs_normal!(φ̇, φ, rng)
    @. φ += phase_speed * sqrt(Δt / 2) * φ̇ # now at t = 0.5

    rhs!(k₂, S̃, parameters)
    @. S̃ = S + Δt * k₂ * 0.5
    rhs!(k₃, S̃, parameters)
    @. S̃ = S + Δt * k₃

    φ_rhs_normal!(φ̇, φ, rng)
    @. φ += phase_speed * sqrt(Δt / 2) * φ̇ # now at t = 1.0

    rhs!(k₄, S̃, parameters)
    @. S += Δt / 6 * (k₁ + 2 * k₂ + 2 * k₃ + k₄)

    t[1] += Δt
    # save output
    #=
    if t[1] > 40
        if i % div(iend, 400) == 0
            global saveindex += 1
            fid["vorticity"][string(saveindex)] = quick_interpolation(ζ)
            fid["moisture"][string(saveindex)] = quick_interpolation(θ)
        end
    end
    =#
    if i == start_index
        θ .= u
    end
    if (i > start_index) && (i % 10 == 0)
        # θ_t[:, :, i - start_index] .= real.(Array(θ))
        # u_t[:, :, i - start_index] .= real.(Array(u))
        # v_t[:, :, i - start_index] .= real.(Array(v))
        if i % 4000 == 0
            # decorrelates after 2000 timesteps
            θ .= u
        end
        push!(θ_t, real.(Array(θ)))
        push!(u_t, real.(Array(u)))
        push!(v_t, real.(Array(v)))
        push!(ζ_t, real.(Array(ζ)))
    end

    θ_min, θ_max = extrema(real.(θ))
    ζ_min, ζ_max = extrema(real.(ζ))
    set_multiline_postfix(iter, "θ_min: $θ_min \nθ_max: $θ_max \nζ_min: $ζ_min \nζ_max: $ζ_max")
end


tmp_θ = zeros(N, N, length(θ_t))
tmp_u = zeros(N, N, length(u_t))
tmp_v = zeros(N, N, length(v_t))
tmp_ζ = zeros(N, N, length(ζ_t))
for i in eachindex(θ_t)
    tmp_θ[:, :, i] .= θ_t[i]
    tmp_u[:, :, i] .= u_t[i]
    tmp_v[:, :, i] .= v_t[i]
    tmp_ζ[:, :, i] .= ζ_t[i]
end
last_index = length(θ_t)

automean = [mean(tmp_u[1:8:end, 1:8:end, i:last_index] .* tmp_u[1:8:end, 1:8:end, 1:last_index-i+1]) for i in 1:2:400]

tmpbool = abs.(tmp_θ[1,1,:] - tmp_u[1,1,:]) .== 0.0
start_indexlist = eachindex(tmpbool)[tmpbool][2:end-1] # remove edge cases
automean2 = Float64[]
for j in 0:2:399
    push!(automean2, mean([mean(tmp_θ[1:8:end, 1:8:end, i+j] .* tmp_u[1:8:end, 1:8:end, i+j]) for i in start_indexlist]))
end

println("The eulerian diffusivity is ", sum(automean .* Δt))
println("The lagrangian diffusivity is ", sum(automean2 .* Δt))
println("The molecular_diffusivity is ", κ)


fig = Figure()
ax = Axis(fig[1,1])
scatter!(ax, automean[1:100])
scatter!(ax, automean2[1:100])
display(fig)

#=
fig = Figure() 
ax = Axis(fig[1,1])
A_ζ = Array(real.(ζ))
tmp = minimum(abs.(extrema(A_ζ)))
heatmap!(ax, A_ζ,  colormap=:balance, colorrange=(-tmp, tmp))
display(fig)

##


=#

fig2 = Figure()

ax = Axis(fig2[1, 1])
A_ζ = Array(real.(ζ))
tmp = minimum(abs.(extrema(A_ζ)))

sl_x = Slider(fig2[2, 1], range=1:length(θ_t), startvalue=1)
o_index = sl_x.value

field = @lift ζ_t[$o_index]
heatmap!(ax, field, colormap=:balance, colorrange=(-tmp, tmp))
display(fig2)