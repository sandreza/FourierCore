using GLMakie, JLD2, FFTW, Statistics, ProgressBars
using FourierCore, FourierCore.Grid, FourierCore.Domain, Printf, LinearAlgebra
include("timestepping_utils.jl")

"""
returns timeseries and score function
# parameters = (; N = 32, Ne = 2^5, ϵ² = 0.004, κ = 1e-3/4, U = 0.02, λ = 4e3)
"""
function allehn_cahn(; parameters = (; N = 32, Ne = 2^2, ϵ² = 0.004, κ = 1e-3/2, λ = 2e3, U = 0.04))
    # model parameters
    ϵ² = parameters.ϵ² # square of noise strength
    κ = parameters.κ # diffusivity
    λ = parameters.λ    # relaxation timescale for double-well
    U = parameters.U   # Velocity

    # numerical parameters 
    N  = parameters.N # number of gridpoints
    Ne = parameters.Ne # number of ensemble members

    # temporal parameters 
    start_time = 100
    end_time = 2000+start_time

    @info "Define domain and operators"
    L = 2π # Domain Size: return time τ = L/U = 157 for default parameters
    Ω = S¹(L)^2 × S¹(1) # Last Domain for "ensembles" 
    grid = FourierGrid((N, N, Ne), Ω, arraytype = Array)
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
    θ = zeros(N, N, Ne) * im + randn(N, N, Ne)
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
    NN = N 
    dt = 1.0/NN
    start_index = start_time * NN
    iterations = end_time * NN 
    pixel_value = zeros(N, N, Ne, iterations - start_index)

    for i in ProgressBar(1:iterations)
        # N^2 comes from fft of white. Could take fft then divide by "two" 
        # the "two" due to imaginary part variance
        rk(rhs, θ, dt)
        noise = real.(sqrt(dt) * ( P⁻¹ * (N^2 * Σhalf .* (randn(N, N, Ne) .+ im * randn(N, N, Ne)))))
        θ .= rk.xⁿ⁺¹ + noise 
        if any(isnan.(θ))
            println("NaN at iteration $i")
            break
        end
        if i > start_index
            pixel_value[:, :, :, i - start_index] .= real.(θ) 
        end
    end

    function score_function(θ)
        nonlinearity = nonlinear_func(θ)
        return  real.(nonlinearity + P⁻¹ * (Σ⁻¹ .* (κ * Δ) .* (P * θ)))
    end

    return pixel_value, score_function
end

function linear_response_function(pixel_value; skip = 32, dt = 1/32)
    N = size(pixel_value)[1]
    Ne = size(pixel_value)[3]
    pv = pixel_value .- mean(pixel_value);
    endindex = size(pv[:, :, :, 1:skip:end])[end]÷2
    ts = skip * dt .* collect(0:endindex-1)
    spatio_temporal_autocov = zeros(N, N, endindex);
    @info "Computing Space-Time Autocorrelation"
    for i in ProgressBar(1:Ne)
        spatio_temporal_autocov .+= real.(ifft(fft(pv[:, :, i, 1:skip:end]) .* ifft(pv[:, :, i, 1:skip:end]))[:, :, 1:endindex]/Ne)
    end
    @info "Computing Covariance Matrix"
    covmat = zeros(N^2, N^2, endindex);
    for k in ProgressBar(1:endindex)
        for i in 1:N^2
            ii = (i-1) % N + 1
            jj = (i-1) ÷ N + 1
            @inbounds covmat[:, i, k] .= circshift(spatio_temporal_autocov[:, :, k], (ii-1, jj-1))[:]
        end
    end
    @info "Inverting Covariance Matrix"
    C⁻¹ = inv(covmat[:, :, 1])
    ##
    @info "Computing Response Function"
    return response_function = [covmat[:, :, i] * C⁻¹ for i in ProgressBar(1:endindex)]
end

function score_response_function(pixel_value, score_function; skip = 32, dt = 1/32)
    N = size(pixel_value)[1]
    Ne = size(pixel_value)[3]
    endindex = size(pixel_value[:, :, :, 1:skip:end])[end]÷2
    score_response_array = zeros(N, N, endindex); 
    score_values = zeros(N, N, Ne, length(1:skip:size(pixel_value)[end]));
    @info "Computing Score Function"
    for (ii, i) in ProgressBar(enumerate(1:skip:size(pixel_value)[end]))
        score_values[:, :, :, ii] .= score_function(pixel_value[:, :, :, i])
    end
    @info "Computing Response Function"
    for i in ProgressBar(1:Ne)   
        score_response_array .+= -real.(ifft(fft(pixel_value[:, :, i, 1:skip:end]) .* ifft(score_values[:, :, i, :]))[:, :, 1:endindex]/Ne/ (N^2/2))
    end
    return score_response_array
end

function hack_score_response_function(pixel_value, score_function; skip = 32, dt = 1/32)
    score_response_array = score_response_function(pixel_value, score_function; skip = skip, dt = dt)
    N, N, endindex = size(score_response_array)
    hack_response = zeros(N^2, N^2, endindex);
    @info "Constructing Hack Matrix"
    for k in ProgressBar(1:endindex)
        for i in 1:N^2
            ii = (i-1) % N + 1
            jj = (i-1) ÷ N + 1
            @inbounds hack_response[:, i, k] .= circshift(score_response_array[:, :, k], (ii-1, jj-1))[:]
        end
    end
    @info "Inverting Hack Matrix"
    C⁻¹ = inv(hack_response[:, :, 1])
    # hack_score_response = [(hack_response[:, :, i] * C⁻¹ + C⁻¹ * hack_response[:, :, i])/2  for i in 1:endindex]
    @info "Constructing Hack Response"
    return [hack_response[:, :, i] * C⁻¹ for i in ProgressBar(1:endindex)]
end
