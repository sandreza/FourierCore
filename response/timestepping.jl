@info "Defining rhs timestepping"
function rhs!(Ṡ, S, t, parameters)
    ζ̇ = view(Ṡ, :, :, :, 1)
    ζ = view(S, :, :, :, 1)

    (; P, P⁻¹, Δ⁻¹, waver, 𝒟ν, 𝒟κ, ∂x, ∂y) = parameters.operators
    (; ψ, x, y, φ, u, v, uζ, vζ, uθ, vθ, ∂ˣζ, ∂ʸζ, ∂ˣθ, ∂ʸθ, ∂ˣuζ, ∂ʸvζ, ∂ˣuθ, ∂ʸvθ, 𝒟θ, 𝒟ζ, sθ, sζ) = parameters.auxiliary
    (; forcing_amplitude, ϵ, ωs) = parameters.constants

    # construct source for vorticity 
    # @. sζ = ψ
    sζ .= waver .* forcing_amplitude .* exp.(im .* φ)
    P⁻¹ * sζ

    P * ζ # in place fft
    # grab stream function from vorticity
    @. ψ = Δ⁻¹ * ζ
    # ∇ᵖψ
    @. u = (∂y * ψ)
    @. v = -1.0 * (∂x * ψ)
    # ∇ζ
    @. ∂ˣζ = ∂x * ζ
    @. ∂ʸζ = ∂y * ζ
    # Dissipation
    @. 𝒟ζ = 𝒟ν * ζ
    # go back to real space 
    P⁻¹ * u
    u .+= 1.0
    P⁻¹ * v
    P⁻¹ * ζ
    P⁻¹ * ∂ˣζ
    P⁻¹ * ∂ʸζ
    P⁻¹ * 𝒟ζ
    # construct conservative form 
    @. uζ = u * ζ
    @. vζ = v * ζ
    # in place fft 
    P * uζ
    P * vζ
    # ∇⋅(u⃗ζ)
    @. ∂ˣuζ = ∂x * uζ
    @. ∂ʸvζ = ∂y * vζ
    # in place ifft 
    P⁻¹ * ∂ˣuζ
    P⁻¹ * ∂ʸvζ

    # rhs
    @. ζ̇ = real((-u * ∂ˣζ - v * ∂ʸζ - ∂ˣuζ - ∂ʸvζ) * 0.5 + 𝒟ζ + sζ)
    @. S = real(S)
    @. Ṡ = real(Ṡ)

    return nothing
end

function step!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, t, parameters)
    rhs!(k₁, S, t, parameters)
    @. S̃ = S + Δt * k₁ * 0.5
    randn!(rng, φ̇)
    t[1] += Δt / 2
    @. φ += phase_speed * sqrt(Δt / 2 * 2) * φ̇ # now at t = 0.5, note the factor of two has been accounted for
    rhs!(k₂, S̃, t, parameters)
    @. S̃ = S + Δt * k₂ * 0.5
    rhs!(k₃, S̃, t, parameters)
    @. S̃ = S + Δt * k₃
    randn!(rng, φ̇)
    t[1] += Δt / 2
    @. φ += phase_speed * sqrt(Δt / 2 * 2) * φ̇ # now at t = 1.0, note the factor of two has been accounted for
    rhs!(k₄, S̃, t, parameters)
    @. S += Δt / 6 * (k₁ + 2 * k₂ + 2 * k₃ + k₄)
    return nothing
end
