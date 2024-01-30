f_amps = [50, 150, 300, 450, 750, 0.1, 1, 10, 0.01]
νs = [sqrt(1e-5 / 2)]
ν_hs = [sqrt(1e-3), sqrt(1e-4), sqrt(1e-2)]
tic = Base.time()

base_name = "off_diagonal_case_"
N = 2^7
N_ens = 2^7 # 2^7
Ns = (N, N, N_ens)

jj = 1
ii = 2
kk = 1
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]


include("timestepping.jl") # independent of everything else

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
tend = 2^6 * scaleit * 2
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
include("diffusivity_kernel_off_diagonal.jl") # tracer is initialized in here