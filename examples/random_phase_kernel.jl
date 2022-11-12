using KernelAbstractions
using CUDAKernels

# only expect performance on the gpu
# keep in mind that ∂x is somehow using global scope information and still remaining okay

@kernel function random_phase_kernel!(field, @Const(A), @Const(𝓀ˣ), @Const(𝓀ʸ), @Const(x), @Const(y), @Const(φ), Nx, Ny)
    i, j = @index(Global, NTuple)

    tmp_sum = zero(eltype(field)) # create temporary array
    xx = x[i]
    yy = y[j]
    for ii in 1:Nx, jj in 1:Ny
        tmp_sum += A[ii, jj] * cos(𝓀ˣ[ii] * xx + 𝓀ʸ[jj] * yy + φ[ii, jj])
    end

    field[i, j] = tmp_sum
end



function stream_function!(field, A, 𝓀ˣ, 𝓀ʸ, x, y, φ; comp_stream=Event(CUDADevice()))
    kernel! = random_phase_kernel!(CUDADevice(), 256)
    Nx = length(𝓀ˣ)
    Ny = length(𝓀ʸ)
    event = kernel!(field, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, Nx, Ny, ndrange=size(field), dependencies=(comp_stream,))
    # event = kernel!(field, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, Nx, Ny, ndrange = size(field))
    # wait(event) here on event = kernel! causes the gpu to hang, need to wait outside
    return event
end

# Expensive on GPU but better on CPU
function random_phase!(field, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
    field .= 0.0
    for i in eachindex(𝓀ˣ), j in eachindex(𝓀ʸ)
        @. field += A[i, j] * cos(𝓀ˣ[i] * x + 𝓀ʸ[j] * y + φ[i, j])
    end
end

function θ_rhs!(θ̇, θ, params)
    #(; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, s, P, P⁻¹, filter) = params
    ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, s, P, P⁻¹, filter = params
    P * ψ # in place fft
    P * θ # in place fft
    # ∇ᵖψ
    @. u = filter * -1.0 * (∂y * ψ)
    @. v = filter * (∂x * ψ)
    # ∇θ
    @. ∂ˣθ = filter * ∂x * θ
    @. ∂ʸθ = filter * ∂y * θ
    @. κΔθ = κ * Δ * θ
    # go back to real space 
    P⁻¹ * ψ
    P⁻¹ * θ
    P⁻¹ * u
    P⁻¹ * v
    P⁻¹ * ∂ˣθ
    P⁻¹ * ∂ʸθ
    P⁻¹ * κΔθ
    # Assemble RHS
    @. θ̇ = -u * ∂ˣθ - v * ∂ʸθ + κΔθ + s
    return nothing
end

function θ_rhs_new!(θ̇, θ, params)
    #(; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, s, P, P⁻¹, filter) = params
    ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, s, P, P⁻¹, filter = params
    event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
    wait(event)
    P * ψ # in place fft
    P * θ # in place fft
    # ∇ᵖψ
    @. u = filter * -1.0 * (∂y * ψ)
    @. v = filter * (∂x * ψ)
    # ∇θ
    @. ∂ˣθ = filter * ∂x * θ
    @. ∂ʸθ = filter * ∂y * θ
    @. κΔθ = κ * Δ * θ
    # go back to real space 
    P⁻¹ * ψ
    P⁻¹ * θ
    P⁻¹ * u
    P⁻¹ * v
    P⁻¹ * ∂ˣθ
    P⁻¹ * ∂ʸθ
    P⁻¹ * κΔθ
    # Assemble RHS
    @. θ̇ = -u * ∂ˣθ - v * ∂ʸθ + κΔθ + s
    return nothing
end

function θ_rhs_symmetric!(θ̇, θ, simulation_parameters)
    (; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, filter) = simulation_parameters
    event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
    wait(event)
    P * ψ # in place fft
    P * θ # in place fft
    # ∇ᵖψ
    @. u = filter * -1.0 * (∂y * ψ)
    @. v = filter * (∂x * ψ)
    # ∇θ
    @. ∂ˣθ = filter * ∂x * θ
    @. ∂ʸθ = filter * ∂y * θ
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
    @. ∂ˣuθ = filter * ∂x * uθ
    @. ∂ʸvθ = filter * ∂y * vθ
    P⁻¹ * ∂ˣuθ
    P⁻¹ * ∂ʸvθ
    # Assemble RHS
    @. θ̇ = -(u * ∂ˣθ + v * ∂ʸθ + ∂ˣuθ +  ∂ʸvθ) * 0.5  + κΔθ + s
    return nothing
end


function θ_rhs_symmetric_zeroth!(θ̇, θ, simulation_parameters)
    (; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, filter, ∂x, ∂y, κ, Δ, κΔθ) = simulation_parameters
    event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
    wait(event)
    P * ψ # in place fft
    P * θ # in place fft
    # ∇ᵖψ
    @. u = filter * -1.0 * (∂y * ψ)
    @. v = filter * (∂x * ψ)
    # ∇θ
    @. ∂ˣθ = filter * ∂x * θ
    @. ∂ʸθ = filter * ∂y * θ
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
    @. ∂ˣuθ = filter * ∂x * uθ
    @. ∂ʸvθ = filter * ∂y * vθ
    P⁻¹ * ∂ˣuθ
    P⁻¹ * ∂ʸvθ
    # Assemble RHS
    @. θ̇ = -(u * ∂ˣθ + v * ∂ʸθ + ∂ˣuθ +  ∂ʸvθ) * 0.5  + κΔθ + u
    return nothing
end

function θ_rhs_symmetric_ensemble!(θ̇, θ, simulation_parameters)
    (; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, uθ, vθ, ∂ˣuθ, ∂ʸvθ, s, s¹, P, P⁻¹, filter) = simulation_parameters
    event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
    wait(event)
    P * ψ # in place fft
    P * θ # in place fft
    # ∇ᵖψ
    @. u = filter * -1.0 * (∂y * ψ)
    @. v = filter * (∂x * ψ)
    # ∇θ
    @. ∂ˣθ = filter * ∂x * θ
    @. ∂ʸθ = filter * ∂y * θ
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
    @. ∂ˣuθ = filter * ∂x * uθ
    @. ∂ʸvθ = filter * ∂y * vθ
    P⁻¹ * ∂ˣuθ
    P⁻¹ * ∂ʸvθ
    # Assemble RHS
    @. θ̇ = -(u * ∂ˣθ + v * ∂ʸθ + ∂ˣuθ + ∂ʸvθ) * 0.5 + κΔθ + u * s¹  + s
    return nothing
end




function θ_rhs_zeroth!(θ̇, θ, params)
    #(; ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, s, P, P⁻¹, filter) = params
    ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣθ, ∂ʸθ, s, P, P⁻¹, filter = params
    P * ψ # in place fft
    P * θ # in place fft
    # ∇ᵖψ
    @. u = filter * -1.0 * (∂y * ψ)
    @. v = filter * (∂x * ψ)
    # ∇θ
    @. ∂ˣθ = filter * ∂x * θ
    @. ∂ʸθ = filter * ∂y * θ
    @. κΔθ = κ * Δ * θ
    # go back to real space 
    P⁻¹ * ψ
    P⁻¹ * θ
    P⁻¹ * u
    P⁻¹ * v
    P⁻¹ * ∂ˣθ
    P⁻¹ * ∂ʸθ
    P⁻¹ * κΔθ
    # Assemble RHS
    @. θ̇ = -u * ∂ˣθ - v * ∂ʸθ + κΔθ + u
    return nothing
end

function θ_rhs_convergent!(θ̇, θ, params)
    ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, u, v, ∂ˣuθ, ∂ʸvθ, s, P, P⁻¹, filter, uᶜ, vᶜ = params
    P * ψ # in place fft
    # ∇ᵖψ
    @. u = filter * -1.0 * (∂y * ψ)
    @. v = filter * (∂x * ψ)
    P⁻¹ * u
    P⁻¹ * v
    # ∇⋅(u⃗ θ) + κΔ
    @. ∂ˣuθ = (u + uᶜ) * θ
    @. ∂ʸvθ = (v + vᶜ) * θ
    P * ∂ˣuθ # in place fft
    P * ∂ʸvθ # in place fft
    P * θ  # in place fft
    @. ∂ˣuθ = ∂x * ∂ˣuθ
    @. ∂ʸvθ = ∂y * ∂ʸvθ
    @. κΔθ = κ * Δ * θ
    # go back to real space 
    P⁻¹ * ∂ˣuθ
    P⁻¹ * ∂ʸvθ
    P⁻¹ * θ
    P⁻¹ * κΔθ
    P⁻¹ * ψ # technically not necessary
    # Assemble RHS
    @. θ̇ = -∂ˣuθ - ∂ʸvθ + κΔθ
    return nothing
end

function φ_rhs!(φ̇, φ, rng)
    rand!(rng, φ̇) # can use randn(rng, φ̇); @. φ̇ *= sqrt(1/12)
    φ̇ .-= 0.5
    return nothing
end

function φ_rhs_normal!(φ̇, φ, rng)
    randn!(rng, φ̇)
    # φ̇ .*= sqrt(1 / 12)
    φ̇ .*= sqrt(2)
    return nothing
end
