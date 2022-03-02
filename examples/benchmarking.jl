

@benchmark random_phase_check!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)


@benchmark begin
    P⁻¹ * ψ
    P⁻¹ * θ
    P⁻¹ * u
    P⁻¹ * v
    P⁻¹ * ∂ˣθ
    P⁻¹ * ∂ʸθ
    P⁻¹ * κΔθ
end

@benchmark begin
    @. u = filter * -1.0 * (∂y * ψ)
    @. v = filter * (∂x * ψ)
    # ∇θ
    @. ∂ˣθ = filter * ∂x * θ
    @. ∂ʸθ = filter * ∂y * θ
    @. κΔθ = κ * Δ * θ
end

@benchmark begin
    φ̇ .= arraytype((rand(size(A)...) .- 0.5))
    @. θ̇ = -u * ∂ˣθ - v * ∂ʸθ + κΔθ + s
end

@benchmark begin
    @. φ += sqrt(Δt) * φ̇
    @. θ += Δt * θ̇
    t[1] += Δt
end

@benchmark begin
    # spectral space representation 
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
    φ̇ .= arraytype((rand(size(A)...) .- 0.5))
    @. θ̇ = -u * ∂ˣθ - v * ∂ʸθ + κΔθ + s
    # Euler step
    @. φ += sqrt(Δt) * φ̇
    @. θ += Δt * θ̇
    t[1] += Δt
end