κ = 1e-3 # diffusivity for scalar
ν = sqrt(1e-5) # raised to the dissipation_power, 1e-5
dissipation_power = 2
ν_h = sqrt(1e-2) # sqrt(1e-2 / 1.5) # raised to the hypoviscocity_power
hypoviscocity_power = 2
f_amp = 300 # forcing amplitude
forcing_amplitude = f_amp * (N / 2^7)^2 # due to FFT nonsense [check if this is true]
ϵ = 0.0    # large scale parameter, 0 means off, 1 means on
ωs = [0.0]    # frequency, 0 means no time dependence
kmax = 7  # filter for forcing
Δt = 1 / 4N # timestep

constants = (; forcing_amplitude=forcing_amplitude, ϵ=ϵ, ωs=ωs)