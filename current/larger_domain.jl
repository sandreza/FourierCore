f_amps = [50, 150, 300, 450, 750, 0.1, 1, 10, 0.01]
νs = [sqrt(1e-5 / 2), sqrt(1e-5)]
ν_hs = [sqrt(1e-3), sqrt(1e-4), sqrt(10^(-2.0))]
tic = Base.time()

jj = 1
ii = 1
kk = 3
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]


filename = "larger_domain_case_filter_" * string(ii) * "_" * string(jj) * "_" * string(kk)
println("---------------------------------")
println("Computing case $filename with f_amp = $f_amp, ν = $(ν^2), ν_h = $(ν_h^2)")
# Initialize the fields, choose domain size
N = 2^7
N_ens = 2^5 # 2^7
Ns = (N * 4, N, N_ens)

# initialize constants
κ = 1e-3 # 1e-3
dissipation_power = 2
hypoviscocity_power = 2

@info "initializing fields"
using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, HDF5, ProgressBars, Statistics
using CUDA
arraytype = CuArray

rng = MersenneTwister(12345)
Random.seed!(12)

phase_speed = 1.0

Ω = S¹(4 * 4π) × S¹(4π) × S¹(1)
grid = FourierGrid(Ns, Ω, arraytype=arraytype)
nodes, wavenumbers = grid.nodes, grid.wavenumbers

# build filter 
x = nodes[1]
y = nodes[2]
kˣ = wavenumbers[1]
kʸ = wavenumbers[2]

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

# prognostic variables
S = arraytype(zeros(ComplexF64, size(ψ)..., 2))
Ṡ = arraytype(zeros(ComplexF64, size(ψ)..., 2))

# auxiliary fields
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

# view into prognostic variables θ and ζ
θ = view(S, :, :, :, 1)
ζ = view(S, :, :, :, 2)
@. θ = 0.0 * sin(3 * x) * sin(3 * y) + -0.1 + 0im
@. ζ = sin(3 * x) * sin(3 * y)

auxiliary = (; ψ, x, y, φ, u, v, uζ, vζ, uθ, vθ, ∂ˣζ, ∂ʸζ, ∂ˣθ, ∂ʸθ, ∂ˣuζ, ∂ʸvζ, ∂ˣuθ, ∂ʸvθ, 𝒟θ, 𝒟ζ, sθ, sζ)

function momentum_filter(S)
    ζ = view(S, :, :, :, 2)
    P⁻¹ * ζ
    tmpfield = abs.(kˣ) .>= 0.5
    ζ .*= tmpfield
    P * ζ
    return nothing
end

function load_psi!(ψ; filename="case_1", directory="/storage5/NonlocalPassiveTracers/Current/")
    fid = h5open(directory * filename * ".hdf5", "r")
    ψ .= arraytype(read(fid["stream function"]))
    close(fid)
    return nothing
end

@info "done initializing fields"

forcing_amplitude = f_amp
ϵ = 0.0
ωs = [0.0]

Δt = 2 / 32 # 8 / N # timestep
scaleit = 2^8 #  2^9
kmax = 25  # filter for forcing
@info "initializing operators"
# operators
∂x = im * kˣ
∂y = im * kʸ
Δ = @. ∂x^2 + ∂y^2

# plan ffts
P = plan_fft!(ψ, (1, 2))
P⁻¹ = plan_ifft!(ψ, (1, 2))

# Dissipation 
Δ = @. ∂x^2 + ∂y^2
Δ⁻¹ = 1 ./ Δ
bools = (!).(isnan.(Δ⁻¹))
Δ⁻¹ .*= bools # hack in the fact that false * NaN = 0

𝒟ν = @. -(-ν_h * Δ⁻¹)^(hypoviscocity_power) - (-ν * Δ)^(dissipation_power)
𝒟κ = @. κ * Δ

# filter for forcing 
# construct waver
kxmax = maximum(kˣ)
kymax = maximum(kʸ)
kxymax = maximum([kxmax, kymax])
kxmax = kymax = kmax
waver = @. (kˣ)^2 + (kʸ)^2 ≤ ((kxmax / 2)^2 + (kymax / 2)^2)
waver = @. (kˣ)^2 + (kʸ)^2 ≤ 0.5 * kxymax^2
waver .*= @. (kˣ != 0.0) .* (kʸ != 0.0)
# waver = @. abs(kˣ) .+ 0 * abs(kʸ) ≤ 2 / 3 * kxmax
# @. waver = waver * (0 * abs(kˣ) .+ 1 * abs(kʸ) ≤ 2 / 3 * kxmax)
# orig_dom = abs.(kˣ .% 0.5) .< eps(1.0)
# @. waver = waver * orig_dom
waver[1, :] .= 1.0
waver[:, 1] .= 1.0
waver[1, 1] = 0.0

# waver[:, floor(Int, N / 2)+1] .= 0.0
# waver[floor(Int, N / 2)+1, :] .= 0.0

operators = (; P, P⁻¹, Δ⁻¹, waver, 𝒟ν, 𝒟κ, ∂x, ∂y)

##
include("timestepping.jl")
##
@show "initializing ensembles"
sθ .= 0.0

tstart = 2^5 * scaleit # start time for gathering statistics
tend = 2^6 * scaleit # simulation endtime 2^9 is 512
mod_index = scaleit # save every other mod index
decorrelation_index = 2^8 * scaleit # how many steps till we reinitialize tracer, for lagrangian decorrelation
decorrelation_index2 = 2^10 * scaleit # how many steps till we reinitialize u₀, for eulerian decorrelation

constants = (; forcing_amplitude=forcing_amplitude, ϵ=ϵ, ωs=ωs)
parameters = (; auxiliary, operators, constants) # auxiliary was defined in initialize_fields.jl

iend = ceil(Int, tend / Δt)
start_index = round(Int, tstart / Δt)
eulerian_list = Float64[]
lagrangian_list = Float64[]
ke_list = Float64[]
tlist = Float64[]
t = [0.0]

t .= 0.0
iter = ProgressBar(1:iend)
for i = iter
    step!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, t, parameters)
    if i == start_index
        θ .= u
        u₀ .= u
    end
    if (i > start_index) && (i % mod_index == 0)
        if i % decorrelation_index == 0
            θ .= u
        end
        if i % decorrelation_index2 == 0
            u₀ .= u
        end
        uu = real(mean(u .* u₀))
        tmpuθ = real(mean(u .* θ))

        push!(eulerian_list, uu)
        push!(lagrangian_list, tmpuθ)
        push!(ke_list, real(mean(u .* u + v .* v)))
        push!(tlist, t[1])

        θ_min, θ_max = extrema(Array(real.(θ))[:])
        ζ_min, ζ_max = extrema(Array(real.(ζ))[:])
        s1 = "θ_min: $θ_min \nθ_max: $θ_max \nζ_min: $ζ_min \nζ_max: $ζ_max"
        s2 = "\nuu : $(uu) \nuθ : $(tmpuθ)"
        set_multiline_postfix(iter, s1 * s2)
    end
end

# lagrangian further ensemble averaging
skipL = round(Int, decorrelation_index / mod_index)
skipL = minimum([skipL, length(lagrangian_list)])
# start_index
# si = Int(maximum([argmax(lagrangian_list) % skipL, skipL]))
si = 1
# end index
ei = floor(Int, (length(lagrangian_list) - si + 1) / skipL)
formatted_lagrangian_list = [lagrangian_list[si+(i-1)*skipL:si+i*skipL-2] for i in 1:ei]

# eulerian further ensemble averaging 
skip = round(Int, decorrelation_index2 / mod_index)
skip = minimum([skip, length(lagrangian_list)])
# start_index
si = Int(maximum([argmax(eulerian_list) % skip, skip]))
si = 1
# end index
ei = floor(Int, (length(eulerian_list) - si + 1) / skip)
formatted_eulerian_list = [eulerian_list[si+(i-1)*skip:si+i*skip-1] for i in 1:ei]

directory = "/storage5/NonlocalPassiveTracers/Current/"
fid = h5open(directory * filename * ".hdf5", "w")
fid["forcing amplitude"] = f_amp
fid["Nx"] = N
fid["Ny"] = N
fid["Nensemble"] = N_ens
fid["nu"] = ν^dissipation_power
fid["nu harmonic power"] = dissipation_power
fid["kappa"] = κ
fid["hypoviscocity"] = ν_h^hypoviscocity_power
fid["hypoviscosity power"] = hypoviscocity_power
fid["vorticity"] = real.(Array(ζ))
P⁻¹ * ψ
fid["stream function"] = real.(Array(ψ))
fid["lagrangian decorrelation"] = mean(formatted_lagrangian_list)
fid["lagrangian decorrelation unprocessesed"] = lagrangian_list
fid["eulerian decorrelation unprocessesed"] = eulerian_list
fid["eulerian decorrelation"] = mean(formatted_eulerian_list)
fid["kinetic energy evolution"] = ke_list
fid["times output decorrelation case"] = tlist
fid["domain size x"] = Ω[1].b - Ω[1].a
fid["domain size y"] = Ω[2].b - Ω[2].a
fid["dt"] = Δt
close(fid)

@show "done initializing ensembles"


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

maxind = minimum([40 * 4, floor(Int, Ns[1] / 4)])
index_choices = 2:maxind # 2:maxind

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

θ .= 0.0 # initialize with zero
t = [0.0]
iend = ceil(Int, tend / Δt)

# new realization of flow
rand!(rng, φ) # between 0, 1
φ .*= 2π # to make it a random phase

θ̄ = arraytype(zeros(ComplexF64, Ns[1], Ns[2], N_ens))

iter = ProgressBar(1:iend)
ke_list = Float64[]
k_list = Vector{Float64}[]
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

        tmp = Array(real.(fft(mean(θ̄, dims=(2, 3))[:]))) ./ (t[1] - tstart) # tmp = real.(fft(Array(mean(θ[:,:,1:10], dims = (2,3)))[:]))
        kxa = Array(kˣ)[:]
        effective_diffusivities = ((Ns[1] / 2) ./ tmp) ./ (kxa .^ 2) .- κ # (((Ns[1] / 2) ./ tmp) .- λ) ./ (kxa .^ 2) .- κ
        effective_diffusivities = effective_diffusivities[index_choices]
        push!(k_list, effective_diffusivities)
    end
end

θ̄ ./= (tend - tstart)
θ̄_A = Array(real.(θ̄))


tmp = Array(real.(fft(mean(θ̄, dims=(2, 3))[:]))) # tmp = real.(fft(Array(mean(θ[:,:,1:10], dims = (2,3)))[:]))
kxa = Array(kˣ)[:]
effective_diffusivities = ((Ns[1] / 2) ./ tmp) ./ (kxa .^ 2) .- κ
effective_diffusivities = effective_diffusivities[index_choices]

# estimate kernel on grid
kernel = real.(fft([0.0, effective_diffusivities..., zeros(65)..., reverse(effective_diffusivities)...]))
kernel = kernel .- mean(kernel[63:65])
kernel = circshift(kernel, floor(Int, Ns[1] / 2))


# save computation
fid = h5open(directory * filename * ".hdf5", "r+")
fid["diffusivity kernel fourier"] = effective_diffusivities
fid["kernel"] = kernel
fid["ensemble mean in fourier space from diffusivity calculation"] = tmp
close(fid)
##
using GLMakie
scatter(effective_diffusivities)
#=
P * ψ;
ζ .= Δ .* ψ;
P⁻¹ * ζ; # initalize stream function and vorticity
ϵ = 1.0    # large scale parameter, 0 means off, 1 means on
ωs = [0.0]    # frequency, 0 means no time dependence

# need to change the parameters and constants every time
constants = (; forcing_amplitude=forcing_amplitude, ϵ=ϵ, ωs=ωs)
parameters = (; auxiliary, operators, constants) # auxiliary was defined in initialize_fields.jl
@. u = (∂y * ψ)
P⁻¹ * u
θ .= u


start_index = floor(Int, tstart / Δt)
sθ .= 0.0
t = [0.0]
iend = ceil(Int, tend / Δt)
# new realization of flow
rand!(rng, φ) # between 0, 1
φ .*= 2π # to make it a random phase

# θ̄ = arraytype(zeros(ComplexF64, N, N, N_ens))

iter = ProgressBar(1:iend)
ke_list = Float64[]
uθ_list = Float64[]
tlist = Float64[]

push!(uθ_list, real(mean(θ .* u)))
push!(ke_list, real(mean(u .* u + v .* v)))
push!(tlist, t[1])
t .= 0.0
for i = iter
    step_filter!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, t, parameters)
    push!(uθ_list, real(mean(θ .* u)))
    push!(tlist, t[1])
    if i % mod_index == 0
        θ_min, θ_max = extrema(real.(θ))
        ζ_min, ζ_max = extrema(real.(ζ))
        s1 = "θ_min: $θ_min \nθ_max: $θ_max \nζ_min: $ζ_min \nζ_max: $ζ_max"
        s2 = "\nuθ   : $(uθ_list[i])"
        set_multiline_postfix(iter, s1 * s2)
    end
    if i > start_index
        # u is already in real space as is θ
        # θ̄ .+= Δt .* θ .* u
    end
    if i % mod_index == 0
        push!(ke_list, real(mean(u .* u + v .* v)))
    end
end

# θ̄ ./= (tend - tstart)
# θ̄_A = Array(real.(θ̄))

@info "done with large scale case"
fid = h5open(directory * filename * ".hdf5", "r+")
fid["large scale effective diffusivity (with time evolution)"] = uθ_list
fid["large scale effective diffusivity times"] = tlist
large_scale = mean(uθ_list[start_index:end])
fid["large scale effective diffusivity"] = large_
=#