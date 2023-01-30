include("timestepping.jl") # independent of everything else

# filename = "case_1" # used in "initialize_ensembles.jl" file
# f_amp = 10 # forcing amplitude
#=
f_amps = [150, 300, 450]
νs = [sqrt(1e-5/2)] # [sqrt(1e-4), sqrt(1e-5 / 2)]
ν_hs = [sqrt(1e-3), sqrt(1e-4)]

ii = 1 
jj = 1
kk = 1
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
defined outside
=#
# for (ii, f_amp) in ProgressBar(enumerate(f_amps))
# for (jj, ν) in enumerate(νs)
# for (kk, ν_h) in enumerate(ν_hs)

filename = "case_" * string(ii) * "_" * string(jj) * "_" * string(kk)
println("---------------------------------")
println("Computing case $filename with f_amp = $f_amp, ν = $(ν^2), ν_h = $(ν_h^2)")
# Initialize the fields, choose domain size
N = 2^7
N_ens = 2^5 # 2^7
Ns = (N, N, N_ens)

include("initialize_fields.jl") # allocates memory for efficiency, defines stream function vorticity etc.

# initialize constants
κ = 1e-3 # diffusivity for scalar
# ν = sqrt(1e-5 / 2) # raised to the dissipation power
dissipation_power = 2
# ν_h = sqrt(1e-3) # raised to the hypoviscocity_power 
hypoviscocity_power = 2

forcing_amplitude = f_amp * (N / 2^7)^2 # due to FFT nonsense [check if this is true]
ϵ = 0.0    # large scale parameter, 0 means off, 1 means on
ωs = [0.0]    # frequency, 0 means no time dependence
Δt = 1 / N # timestep
t = [0.0]  # time
kmax = 30  # filter for forcing

# initalize the operators
include("initialize_operators.jl")

# initialize ensembles and gather both eulerian and lagrangian decorrelations
scaleit = 2^3
tstart = 2^5 * scaleit # start time for gathering statistics
tend = 2^6 * scaleit # simulation endtime 2^9 is 512

mod_index = 2^3 # save every other mod index
decorrelation_index = 2^11 # how many steps till we reinitialize tracer, for lagrangian decorrelation
decorrelation_index2 = 2^13 # how many steps till we reinitialize u₀, for eulerian decorrelation

constants = (; forcing_amplitude=forcing_amplitude, ϵ=ϵ, ωs=ωs)
parameters = (; auxiliary, operators, constants) # auxiliary was defined in initialize_fields.jl
include("initialize_ensembles.jl")

## Compute effective diffusivities
# start gathering statistics at tstart and the simulation at tend
# 2^8 is 256
scaleit = 2^3
tstart = 2^5 * scaleit
tend = 2^6 * scaleit
load_psi!(ψ; filename=filename) # was defined in the initalize fields file
P * ψ;
ζ .= Δ .* ψ;
P⁻¹ * ζ; # initalize stream function and vorticity

constants = (; forcing_amplitude=forcing_amplitude, ϵ=ϵ, ωs=ωs)
parameters = (; auxiliary, operators, constants) # auxiliary was defined in initialize_fields.jl
include("diffusivity_kernel.jl") # tracer is initialized in here

# save computation
fid = h5open(directory * filename * ".hdf5", "r+")
fid["diffusivity kernel fourier"] = effective_diffusivities
fid["kernel"] = kernel
close(fid)

## Large scale case 
scaleit = 2^3
tstart = 2^5 * scaleit
tend = 2^6 * scaleit
load_psi!(ψ; filename=filename) # was defined in the initalize fields file
P * ψ;
ζ .= Δ .* ψ;
P⁻¹ * ζ; # initalize stream function and vorticity
ϵ = 1.0    # large scale parameter, 0 means off, 1 means on
ωs = [0.0]    # frequency, 0 means no time dependence

# need to change the parameters and constants every time
constants = (; forcing_amplitude=forcing_amplitude, ϵ=ϵ, ωs=ωs)
parameters = (; auxiliary, operators, constants) # auxiliary was defined in initialize_fields.jl
include("large_scale.jl")
fid = h5open(directory * filename * ".hdf5", "r+")
fid["large scale effective diffusivity (with time evolution)"] = uθ_list
fid["large scale effective diffusivity times"] = tlist
large_scale = mean(uθ_list[start_index:end])
fid["large scale effective diffusivity"] = large_scale
close(fid)

## Large scale time dependent case
scaleit = 2^3
tstart = 2^5 * scaleit
tend = 2^6 * scaleit
load_psi!(ψ; filename=filename) # was defined in the initalize fields file
P * ψ;
ζ .= Δ .* ψ;
P⁻¹ * ζ; # initalize stream function and vorticity
ϵ = 1.0    # large scale parameter, 0 means off, 1 means on
Ts = [2^i for i in [0, 1, 2, 3, 4, 5, 6, 7]]    # power of two for convience
ωs = [2π/T for T in Ts]   # frequency, 0 means no time dependence

# need to change the parameters and constants every time
constants = (; forcing_amplitude=forcing_amplitude, ϵ=ϵ, ωs=ωs)
parameters = (; auxiliary, operators, constants) # auxiliary was defined in initialize_fields.jl
include("large_scale.jl")
fid = h5open(directory * filename * ".hdf5", "r+")
fid["time dependent large scale effective diffusivity"] = uθ_list
fid["time dependent large scale effective diffusivity times"] = tlist
fid["time dependent large scale angular frequencies"] = ωs
fid["time dependent large scale periods"] = Ts
close(fid)

# end
# end
# end

#=
include("diffusivity_kernel.jl")
include("large_scale.jl")
include("large_scale_time_dependent.jl")
=#