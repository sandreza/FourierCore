κ = 1e-3 # diffusivity for scalar
ν = sqrt(1e-5 / 2) # raised to the dissipation_power
dissipation_power = 2
ν_h = sqrt(1e-4) # sqrt(1e-3) # raised to the hypoviscocity_power
hypoviscocity_power = 2
f_amp = 300 # forcing amplitude
forcing_amplitude = f_amp * (N / 2^7)^2 # due to FFT nonsense [check if this is true]
ϵ = 0.0    # large scale parameter, 0 means off, 1 means on
ωs = [0.0]    # frequency, 0 means no time dependence
Δt = 1 / N # timestep
t = [0.0]  # time
kmax = 30  # filter for forcing

constants = (; forcing_amplitude=forcing_amplitude, ϵ=ϵ, ω=ω)