

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




function θ_rhs_symmetric_compressible!(θ̇, θ, simulation_parameters)
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
    @. θ̇ = -(u * ∂ˣθ + v * ∂ʸθ + ∂ˣuθ + ∂ʸvθ) * 0.5 + κΔθ + s
    return nothing
end