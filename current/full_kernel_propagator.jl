f_amps = [50, 150, 300, 450, 750, 0.1, 1, 10, 0.01]
νs = [sqrt(1e-5 / 2)]
ν_hs = [sqrt(1e-3), sqrt(1e-4), sqrt(1e-2)]
tic = Base.time()

base_name = "propagator_"
N = 2^7
N_ens = 2^3 # 2^7
Ns = (N, N, N_ens)

ii = 3 # forcing
kk = 1 # hypo
jj = 1 # hyper
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
Δt = 1 / 2N # timestep
scaleit = 2^4 # 2^4 * 2

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
ϵ = 0.0    # large scale parameter, 0 means off, 1 means on
ωs = [0.0]
constants = (; forcing_amplitude=forcing_amplitude, ϵ=ϵ, ωs=ωs)
parameters = (; auxiliary, operators, constants) # auxiliary was defined in initialize_fields.jl

# just use rhs 
# set  source .= sum(advection , dims = 3)
# set initial condition to u(x = 0, y) δ(x), e.g. u[1, :, :]
# Initialize ensemble and get initial conditions

tend = 10
iend = ceil(Int, tend / Δt)
kernel = zeros(N, iend)

numloops = 10
for kk in ProgressBar(1:numloops)
    rhs!(Ṡ, S, t, parameters) # to calculate u
    t = [0.0]
    θ .= 0
    θ[1, :, : ] .= u[1, : , : ]
    rhs!(Ṡ, S, t, parameters)
    @. sθ = (-u * ∂ˣθ - v * ∂ʸθ - ∂ˣuθ - ∂ʸvθ) * 0.5
    sθ .= -mean(sθ, dims = 3)
    θ .= 0
    θ[1, :, : ] .= real.(u[1, : , : ])

    for i in ProgressBar(1:iend )
        kernel[:, i] .+= circshift(Array(real.(mean(u .* θ, dims = (2, 3)))), 64) / numloops
        step!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, t, parameters)
        @. sθ = (-u * ∂ˣθ - v * ∂ʸθ - ∂ˣuθ - ∂ʸvθ) * 0.5
        sθ .= -mean(sθ, dims = 3)
    end
end


fid["space time kernel"] = kernel
fid["space time kernel timelist"] = collect(0:iend-1) * Δt
#  tmplist = [maximum(abs.(kernel[:, i])) for i in 1:iend]
# [mean(abs.(kernel[:, i])) for i in 1:iend]