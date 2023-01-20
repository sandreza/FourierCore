using KernelAbstractions
using CUDAKernels
# only expect performance on the gpu
@kernel function stirring_kernel!(field, @Const(A), @Const(𝓀ˣ), @Const(𝓀ʸ), @Const(x), @Const(y), @Const(φ), Nx, Ny)
    i, j = @index(Global, NTuple)

    tmp_sum = zero(eltype(field)) # create temporary array
    xx = x[i]
    yy = y[j]
    
    λ = 30.0
    for ii in 1:12
        tmp_sum += A[ii, 1] * (-1)^ii * exp(-λ * (cos(0.5 * (xx - φ[ii, 1]))^2 + cos(0.5 * (yy - φ[ii, 2]))^2))
    end

    field[i, j] = tmp_sum
end


function stream_function!(field, A, 𝓀ˣ, 𝓀ʸ, x, y, φ; comp_stream=Event(CUDADevice()))
    kernel! = stirring_kernel!(CUDADevice(), 256)
    Nx = length(𝓀ˣ)
    Ny = length(𝓀ʸ)
    event = kernel!(field, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, Nx, Ny, ndrange=size(field), dependencies=(comp_stream,))

    # event = kernel!(field, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, Nx, Ny, ndrange = size(field))
    # wait(event) here on event = kernel! causes the gpu to hang, need to wait outside
    return event
end

function θ_rhs_symmetric!(θ̇, θ, simulation_parameters)
    (; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ) = simulation_parameters
    event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
    wait(event)
    P * ψ # in place fft
    P * θ # in place fft
    # ∇ᵖψ
    @. u = -1.0 * (∂y * ψ)
    @. v = (∂x * ψ)
    # ∇θ
    @. ∂ˣθ = ∂x * θ
    @. ∂ʸθ = ∂y * θ
    @. κΔθ = κ * Δ * θ
    # go back to real space 
    P⁻¹ * ψ
    P⁻¹ * θ
    P⁻¹ * u
    P⁻¹ * v
    P⁻¹ * ∂ˣθ
    P⁻¹ * ∂ʸθ
    P⁻¹ * κΔθ
    # compute u * θ and v * θ take derivative and come back
    @. uθ = u * θ
    @. vθ = v * θ
    P * uθ
    P * vθ
    @. ∂ˣuθ = ∂x * uθ
    @. ∂ʸvθ = ∂y * vθ
    P⁻¹ * ∂ˣuθ
    P⁻¹ * ∂ʸvθ
    # Assemble RHS
    @. θ̇ = real(-(u * ∂ˣθ + v * ∂ʸθ + ∂ˣuθ + ∂ʸvθ) * 0.5 + κΔθ + s)
    return nothing
end


function θ_rhs_symmetric_zeroth!(θ̇, θ, simulation_parameters)
    (; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, ∂x, ∂y, κ, Δ, κΔθ) = simulation_parameters
    event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
    wait(event)
    P * ψ # in place fft
    P * θ # in place fft
    # ∇ᵖψ
    @. u = -1.0 * (∂y * ψ)
    @. v = (∂x * ψ)
    # ∇θ
    @. ∂ˣθ = ∂x * θ
    @. ∂ʸθ = ∂y * θ
    @. κΔθ = κ * Δ * θ
    # go back to real space 
    P⁻¹ * ψ
    P⁻¹ * θ
    P⁻¹ * u
    P⁻¹ * v
    P⁻¹ * ∂ˣθ
    P⁻¹ * ∂ʸθ
    P⁻¹ * κΔθ
    # compute u * θ and v * θ take derivative and come back
    @. uθ = u * θ
    @. vθ = v * θ
    P * uθ
    P * vθ
    @. ∂ˣuθ = ∂x * uθ
    @. ∂ʸvθ = ∂y * vθ
    P⁻¹ * ∂ˣuθ
    P⁻¹ * ∂ʸvθ
    # Assemble RHS
    @. θ̇ = -(u * ∂ˣθ + v * ∂ʸθ + ∂ˣuθ + ∂ʸvθ) * 0.5 + κΔθ + u
    return nothing
end

function θ_rhs_symmetric_ensemble!(θ̇, θ, simulation_parameters)
    (; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, s¹, P, P⁻¹) = simulation_parameters
    event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
    wait(event)
    P * ψ # in place fft
    P * θ # in place fft
    # ∇ᵖψ
    @. u = -1.0 * (∂y * ψ)
    @. v = (∂x * ψ)
    # ∇θ
    @. ∂ˣθ = ∂x * θ
    @. ∂ʸθ = ∂y * θ
    @. κΔθ = κ * Δ * θ
    # go back to real space 
    P⁻¹ * ψ
    P⁻¹ * θ
    P⁻¹ * u
    P⁻¹ * v
    P⁻¹ * ∂ˣθ
    P⁻¹ * ∂ʸθ
    P⁻¹ * κΔθ
    # compute u * θ and v * θ take derivative and come back
    @. uθ = u * θ
    @. vθ = v * θ
    P * uθ
    P * vθ
    @. ∂ˣuθ = ∂x * uθ
    @. ∂ʸvθ = ∂y * vθ
    P⁻¹ * ∂ˣuθ
    P⁻¹ * ∂ʸvθ
    # Assemble RHS
    @. θ̇ = -(u * ∂ˣθ + v * ∂ʸθ + ∂ˣuθ + ∂ʸvθ) * 0.5 + κΔθ + u * s¹ + s
    return nothing
end

function φ_rhs_normal!(φ̇, φ, rng)
    randn!(rng, φ̇)
    φ̇ .*= sqrt(2)
    return nothing
end
