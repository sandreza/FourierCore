include("timestepping.jl") # independent of everything else

filename = "case_1" # used in "initialize_ensembles.jl" file

# Initialize the fields, choose domain size
N = 2^7
N_ens = 2^2 # 2^7
Ns = (N, N, N_ens)

include("initialize_fields.jl") # allocates memory for efficiency, defines stream function vorticity etc.

# initialize constants
κ = 1e-3 # diffusivity for scalar
ν = sqrt(1e-5 / 2) # raised to the dissipation power
dissipation_power = 2
ν_h = sqrt(1e-3) # raised to the hypoviscocity_power 
hypoviscocity_power = 2
f_amp = 300 # forcing amplitude
forcing_amplitude = f_amp * (N / 2^7)^2 # due to FFT nonsense [check if this is true]
ϵ = 0.0    # large scale parameter, 0 means off, 1 means on
ω = 0.0    # frequency, 0 means no time dependence
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

constants = (; forcing_amplitude=forcing_amplitude, ϵ=ϵ, ω=ω)
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

constants = (; forcing_amplitude=forcing_amplitude, ϵ=ϵ, ω=ω)
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
ω = 0.0    # frequency, 0 means no time dependence

# need to change the parameters and constants every time
constants = (; forcing_amplitude=forcing_amplitude, ϵ=ϵ, ω=ω)
parameters = (; auxiliary, operators, constants) # auxiliary was defined in initialize_fields.jl
include("large_scale.jl")
fid = h5open(directory * filename * ".hdf5", "r+")
fid["large scale effective diffusivity (with time)"] = uθ_list
fid["large scale effective diffusivity times"] = tlist
fid["large scale effective diffusivity"] = mean(uθ_list[start_index:end])
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
T = 2^5    # power of two for convience
ω = 2π/T   # frequency, 0 means no time dependence

# need to change the parameters and constants every time
constants = (; forcing_amplitude=forcing_amplitude, ϵ=ϵ, ω=ω)
parameters = (; auxiliary, operators, constants) # auxiliary was defined in initialize_fields.jl
include("large_scale.jl")
fid = h5open(directory * filename * ".hdf5", "r+")
fid["time dependent large scale effective diffusivity"] = uθ_list
fid["time dependent large scale effective diffusivity times"] = tlist
fid["time dependent large scale angular frequencies"] = ω
close(fid)


#=
include("diffusivity_kernel.jl")
include("large_scale.jl")
include("large_scale_time_dependent.jl")
=#