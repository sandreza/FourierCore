
f_amps = [50, 150, 300, 450, 750, 0.1, 1, 10, 0.01]
νs = [sqrt(1e-5 / 2)]
ν_hs = [sqrt(1e-3), sqrt(1e-4), sqrt(1e-2)]
tic = Base.time()

base_name = "new_general_case_"
N = 2^7
N_ens = 2^5 # 2^7
Ns = (N, N, N_ens)

ii = 3
jj = 1
kk = 1
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
toc = Base.time()
println("Elapsed time: ", (toc - tic) / (60 * 60), " hours")

include("timestepping.jl")
include("timestepping_2.jl") # independent of everything else

filename = base_name * string(ii) * "_" * string(jj) * "_" * string(kk)
println("---------------------------------")
println("Computing case $filename with f_amp = $f_amp, ν = $(ν^2), ν_h = $(ν_h^2)")
# Initialize the fields, choose domain size

include("initialize_fields.jl") # allocates memory for efficiency, defines stream function vorticity etc.

# initialize constants
κ = 1e-3 # 1e-3 # diffusivity for scalar: OVERWRITTEN JUST BELOW for f_amp < 0.2
# ν = sqrt(1e-5 / 2) # raised to the dissipation power
dissipation_power = 2
# ν_h = sqrt(1e-3) # raised to the hypoviscocity_power 
hypoviscocity_power = 2

forcing_amplitude = f_amp * (Ns[1] / 2^7)^2 # due to FFT nonsense [check if this is true]
ϵ = 0.0    # large scale parameter, 0 means off, 1 means on
ωs = [0.0]    # frequency, 0 means no time dependence
if f_amp > 10
    Δt = 1 / 2N # timestep
    scaleit = 2^4 # 2^4 * 2
elseif 1 < f_amp < 10 + 1
    Δt = 1 / N # timestep
    scaleit = 2^5
elseif 0.1 < f_amp < 1 + 1
    Δt = 4 / N  # timestep
    scaleit = 2^7
elseif 0.05 < f_amp < 0.2
    # κ = 1e-4
    Δt = 16 / N # timestep
    scaleit = 2^5 * 2^2 * 2^2
elseif f_amp < 0.05
    # κ = 1e-4
    Δt = 16 / N # timestep
    scaleit = 2^5 * 4 * 4
else
    Δt = 1 / 2N # timestep
    scaleit = 2^3
end
@info "Δt is $Δt"
t = [0.0]  # time

# initalize the operators
include("initialize_operators.jl")

# initialize ensembles and gather both eulerian and lagrangian decorrelations
tstart = 2^5 * scaleit # start time for gathering statistics
tend = 2^6 * scaleit # simulation endtime 2^9 is 512
println("The end time for the ensemble initialization is $tend")

mod_index = scaleit # save every other mod index
decorrelation_index = 2^8 * scaleit # how many steps till we reinitialize tracer, for lagrangian decorrelation
decorrelation_index2 = 2^10 * scaleit # how many steps till we reinitialize u₀, for eulerian decorrelation

constants = (; forcing_amplitude=forcing_amplitude, ϵ=ϵ, ωs=ωs)
parameters = (; auxiliary, operators, constants) # auxiliary was defined in initialize_fields.jl
include("initialize_ensembles.jl")


## Compute effective diffusivities
# start gathering statistics at tstart and the simulation at tend
# 2^8 is 256
tstart = 2^5 * scaleit
tend = 2^6 * scaleit
load_psi!(ψ; filename=filename) # was defined in the initalize fields file
P * ψ;
ζ .= Δ .* ψ;
P⁻¹ * ζ; # initalize stream function and vorticity
if 0.05 < f_amp < 0.2
    Δt = 16 / N # timestep
    tstart = 2^5 * scaleit * 2^2
    tend = 2^6 * scaleit * 2^2
    @info "changing timestep to $Δt"
end
constants = (; forcing_amplitude=forcing_amplitude, ϵ=ϵ, ωs=ωs)
parameters = (; auxiliary, operators, constants) # auxiliary was defined in initialize_fields.jl


@info "diffusivity kernel"
using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2
# using GLMakie, HDF5
using ProgressBars
rng = MersenneTwister(1234)
Random.seed!(123456789)


maxind = minimum([40, floor(Int, N[1] / 2)])
index_choices = 2:maxind

start_index = floor(Int, tstart / Δt)

sθ .= 0.0
θ .= 0

for index_choice in ProgressBar(index_choices)
    kᶠ = kˣ[index_choice]
    @. θ += cos(kᶠ * x) / (kᶠ)^2 / κ # scaling so that source is order 1
end
P * θ # in place fft
@. 𝒟θ = 𝒟κ * θ
P⁻¹ * 𝒟θ # in place fft
P⁻¹ * θ # in place fft
sθ .+= -𝒟θ # add to source

t = [0.0]
iend = ceil(Int, tend / Δt)

# new realization of flow
rand!(rng, φ) # between 0, 1
φ .*= 2π # to make it a random phase

θ̄ = arraytype(zeros(ComplexF64, N, N, N_ens))

iter = ProgressBar(1:iend)
ke_list = Float64[]
t .= 0.0
for i = iter
    step!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, t, parameters)
    if i % mod_index == 0
        θ_min, θ_max = extrema(real.(θ))
        ζ_min, ζ_max = extrema(real.(ζ))
        set_multiline_postfix(iter, "θ_min: $θ_min \nθ_max: $θ_max \nζ_min: $ζ_min \nζ_max: $ζ_max")
    end
    if i > start_index
        θ̄ .+= Δt .* θ
    end
    if i % mod_index == 0
        push!(ke_list, real(mean(u .* u + v .* v)))
    end
end

θ̄ ./= (tend - tstart)
θ̄_A = Array(real.(θ̄))

tmpsave = Array(real.(mean(θ̄, dims=2)))[:, 1, :]
tmp = Array(real.(fft(mean(θ̄, dims=(2, 3))[:]))) # tmp = real.(fft(Array(mean(θ[:,:,1:10], dims = (2,3)))[:]))
kxa = Array(kˣ)[:]
effective_diffusivities = ((Ns[1] / 2) ./ tmp) ./ (kxa .^ 2) .- κ # (((N[1] / 2) ./ tmp) .- λ) ./ (kxa .^ 2) .- κ
effective_diffusivities = effective_diffusivities[index_choices]

# estimate kernel on grid
kernel = real.(fft([0.0, effective_diffusivities..., zeros(65)..., reverse(effective_diffusivities)...]))
kernel = kernel .- mean(kernel[63:65])
kernel = circshift(kernel, 64)


# save computation
fid = h5open(directory * filename * ".hdf5", "r+")
fid["diffusivity kernel fourier"] = effective_diffusivities
fid["kernel"] = kernel
fid["temporal mean and y spatial mean of tracer"] = tmpsave
fid["kinetic energy evolution during kernel calculation"] = ke_list
close(fid)


# need to change the parameters and constants every time


tstart = 2^5 * scaleit
tend = 2^6 * scaleit
# load_psi!(ψ; filename=filename) # was defined in the initalize fields file
# P * ψ;
# ζ .= Δ .* ψ;
# P⁻¹ * ζ; # initalize stream function and vorticity
ϵ = 1.0    # large scale parameter, 0 means off, 1 means on
Ts = [2^i for i in [0, 1, 2, 3, 4, 5, 6, 7]]    # power of two for convience
ωs = [2π / T for T in Ts]   # frequency, 0 means no time dependence

rng = MersenneTwister(1234)
Random.seed!(123456789)

#=
scaleit = 2^3
tstart = 2^5 * scaleit
tend = 2^6 * scaleit
defined outside script 
=#
maxind = minimum([40, floor(Int, Ns[1] / 2)])
index_choices = 2:maxind
#=
forcing_amplitude = 300.0
ϵ = 1.0
ω = 0.0
constants = (; forcing_amplitude=forcing_amplitude, ϵ=ϵ, ω=ω)
parameters = (; auxiliary, operators, constants)
defined outsdie script
=#

# initialize tracer with velocity field
@. u = (∂y * ψ)
P⁻¹ * u
θ .= u


start_index = floor(Int, tstart / Δt)
t = [0.0]
iend = ceil(Int, tend / Δt)
# new realization of flow
rand!(rng, φ) # between 0, 1
φ .*= 2π # to make it a random phase

# θ̄ = arraytype(zeros(ComplexF64, N, N, N_ens))

iter = ProgressBar(1:iend)
ke_list = Float64[]
tlist = Float64[]


push!(ke_list, real(mean(u .* u + v .* v)))
push!(tlist, t[1])
t .= 0.0

sθ .= 0.0
for index_choice in ProgressBar(index_choices)
    kᶠ = kˣ[index_choice]
    @. θ += cos(kᶠ * x) / (kᶠ)^2 / κ # scaling so that source is order 1
end
P * θ # in place fft
@. 𝒟θ = 𝒟κ * θ
P⁻¹ * 𝒟θ # in place fft
P⁻¹ * θ # in place fft
sθ .+= -𝒟θ # add to source

# need to change the parameters and constants every time
constants = (; forcing_amplitude=forcing_amplitude, ϵ=ϵ, ωs=ωs)
parameters = (; auxiliary, operators, constants) # auxiliary was defined in initialize_fields.jl

# uθ_list = Float64[]
# uθ_list = Vector{Float64}[]
uθ_list = zeros(128, iend)
∇θ_list = zeros(128, iend)
iter = ProgressBar(1:iend)
for i = iter
    step_general!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, t, parameters)
    # push!(uθ_list, Array(real(mean(θ .* u, dims=(2, 3))))[:])
    uθ_list[:, i] .= Array(real(mean(θ .* u, dims=(2, 3))))[:]
    ∇θ_list[:, i] .= Array(real(mean(∂ˣθ, dims=(2, 3))))[:]
    push!(tlist, t[1])
    if i % mod_index == 0
        θ_min, θ_max = extrema(real.(θ))
        ζ_min, ζ_max = extrema(real.(ζ))
        s1 = "θ_min: $θ_min \nθ_max: $θ_max \nζ_min: $ζ_min \nζ_max: $ζ_max"
        # s2 = "\nuθ   : $(uθ_list[i])"
        # set_multiline_postfix(iter, s1 * s2)
        set_multiline_postfix(iter, s1 )
    end
    if i > start_index
        # u is already in real space as is θ
        # θ̄ .+= Δt .* θ .* u
    end
    if i % mod_index == 0
        push!(ke_list, real(mean(u .* u + v .* v)))
    end
end


fid = h5open(directory * filename * ".hdf5", "r+")
fid["time dependent large scale effective diffusivity"] = uθ_list
fid["time dependent large scale effective diffusivity times"] = tlist
fid["time dependent large scale angular frequencies"] = ωs
fid["time dependent large scale periods"] = Ts
##
reduced_fluxes = copy(uθ_list[:, 2^17+1:end])
reduced_gradients = copy(∇θ_list[:, 2^17+1:end])
flux_gradient_fourier =  fft(reduced_fluxes) ./ fft(reduced_gradients)
# indices = tend ./ Ts
indices = round.(Int, tend ./ Ts .+ 1)
# fft(reduced_fluxes, 2)[:, reverse(indices)]
fourier_fluxes = fft(fft(reduced_fluxes, 2)[:, reverse(indices)], 1)
fourier_gradients = fft(fft(reduced_gradients, 2)[:, reverse(indices)], 1)
flux_gradient_fourier = fourier_fluxes ./ fourier_gradients

fid["time dependent fluxes"] = reduced_fluxes
fid["time dependent gradients"] = reduced_gradients 
fid["time dependent effective diffusivity"] = flux_gradient_fourier 
fid["time dependent frequences"] = reverse(ωs)
fid["time dependent periods"] = reverse(Ts)
fid["time dependent fluxes times"] = tlist[2^17+1:end]
fid["time dependent wavenumbers"] = Array(kˣ)[:]
fid["time dependent wavenumber indices to trust"] = collect(index_choices)

close(fid)
##
wn = Array(kˣ)[:]
fig = Figure()
ax = Axis(fig[1,1])
for i in 1:7
    lines!(ax, wn[index_choices], -real.(flux_gradient_fourier[index_choices, i]), label = string(Ts[i]))
end
ax.legend = true

axislegend(ax, position=:rt, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=100, labelsize=40)