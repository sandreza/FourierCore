using GLMakie
using  JLD2, FFTW, Statistics, ProgressBars
using FourierCore, FourierCore.Grid, FourierCore.Domain, Printf, LinearAlgebra
using Random, HDF5, LinearAlgebra, CUDA
include("timestepping_utils.jl")
# include("score_function/timestepping_utils.jl")

"""
returns timeseries
# parameters = (; N = 32, Ne = 2^5, ϵ² = 0.004, κ = 1e-3/4, U = 0.02, λ = 4e3)
"""
function allen_cahn_timeseries(; parameters = (; N = 32, Ne = 2^2, ϵ² = 0.004, κ = 1e-3/2, λ = 2e3, U = 0.04), dt_scale = 1, skip = 32, start_time = 1000, end_time = 2000, array = Array)
    # model parameters
    ϵ² = parameters.ϵ² # square of noise strength
    κ = parameters.κ # diffusivity
    λ = parameters.λ    # relaxation timescale for double-well
    U = parameters.U   # Velocity

    # numerical parameters 
    N  = parameters.N # number of gridpoints
    Ne = parameters.Ne # number of ensemble members

    # temporal parameters 

    @info "Define domain and operators"
    L = 2π # Domain Size: return time τ = L/U = 157 for default parameters
    Ω = S¹(L)^2 × S¹(1) # Last Domain for "ensembles" 
    grid = FourierGrid((N, N, Ne), Ω, arraytype = array)
    nodes, wavenumbers = grid.nodes, grid.wavenumbers
    kx, ky, kz = wavenumbers
    ∂x = im * kx
    ∂y = im * ky
    Δ = @. ∂x^2 + ∂y^2

    # correlated noise
    Σ = real.(ϵ² ./ (1 .- Δ)) 
    Σ⁻¹ = 1 ./ Σ
    Σhalf = sqrt.(Σ)

    @info "Allocate Arrays"
    θ = array(zeros(N, N, Ne) * im + randn(N, N, Ne))
    P = plan_fft(θ, (1, 2))
    P⁻¹ = plan_ifft(θ, (1, 2))
    θ = real.(ifft(Σ .* fft(θ)))
    θ ./= maximum(real.(θ))
    rk = RungeKutta4(θ) 

    θ .= real.(ifft(Σ .* fft(θ)))
    θ ./= maximum(real.(θ))

    @info "Define Model"
    # Define Model: closure on λ
    function nonlinear_func(θ)
        nonlinearity = @. λ * (1 - θ^2) * θ 
        return nonlinearity
    end
    function rhs(θ)
        nonlinearity = nonlinear_func(θ)
        return real.(P⁻¹ * (Σ .* (P  * nonlinearity) +  (κ * Δ .- U * ∂x ) .* (P * (θ))) )
    end
    
    @info "Timestepping"
    NN = N * dt_scale
    dt = 1.0/NN
    start_index = start_time * NN
    iterations = end_time * NN 
    ##
    j_global = 0
    for i in ProgressBar(1:iterations)
        if (i > start_index) && ((i-1) % skip == 0)
            j_global += 1
        end
    end
    ##
    pixel_value = zeros(N, N, Ne, j_global)
    j_global = 0
    for i in ProgressBar(1:iterations)
        # The noise could also be implemented as 
        # noise = (√N)^2 * (P⁻¹ * (Σhalf .* (P * (randn(N, N, Ne)))
        # the first term is comes from the spatial delta function in two dimensions
        # e.g. ⟨ξ(x,y, t) ξ(x', y', t')⟩ = δ(x-x') δ(y-y') δ(t-t')
        # The latter is the fft of white noise 
        # the "two" due to imaginary part variance
        rk(rhs, θ, dt)
        noise = real.(sqrt(dt) * ( P⁻¹ * (N^2 * Σhalf .* array(randn(N, N, Ne) .+ im * randn(N, N, Ne)))))
        θ .= rk.xⁿ⁺¹ + noise 
        if (i%10 == 0)
            if any(isnan.(θ))
                println("NaN at iteration $i")
                break
            end
        end
        if (i > start_index) && ((i-1) % skip == 0)
            j_global += 1
            pixel_value[:, :, :, j_global] .= Array(real.(θ) )
        end
    end

    return pixel_value
end

##
# Strong Nonlinearity
Random.seed!(1234)
parameters = (; N = 128*8, Ne = 2^0, ϵ² = 0.004, κ = 1e-3/4, U = 0.02, λ = 4e3)
pv = allen_cahn_timeseries(; parameters, dt_scale = 1, skip = 128, start_time = 1000, end_time = 1250, array = CuArray)

##
fig = Figure()
ax = Axis(fig[1, 1])
heatmap!(ax, pv[:, :, 1, end], colormap = :balance, colorrange = (-2,2))
hidedecorations!(ax)
ax = Axis(fig[2, 1])
heatmap!(ax, pv[:, :, 1, 1], colormap = :balance, colorrange = (-2,2))
hidedecorations!(ax)
ax2 = Axis(fig[1,2])
hist!(ax2, pv[:, :, 1, end][:], colormap = :balance, colorrange = (-2,2), bins = 100, normalization = :pdf)
ax2 = Axis(fig[2,2])
hist!(ax2, pv[:], colormap = :balance, colorrange = (-2,2), bins = 100, normalization = :pdf)
hidedecorations!(ax)
display(fig)