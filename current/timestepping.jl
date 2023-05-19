@info "Defining rhs timestepping"
function rhs!(Ṡ, S, t, parameters)
    θ̇ = view(Ṡ, :, :, :, 1)
    ζ̇ = view(Ṡ, :, :, :, 2)
    θ = view(S, :, :, :, 1)
    ζ = view(S, :, :, :, 2)

    (; P, P⁻¹, Δ⁻¹, waver, 𝒟ν, 𝒟κ, ∂x, ∂y) = parameters.operators
    (; ψ, x, y, φ, u, v, uζ, vζ, uθ, vθ, ∂ˣζ, ∂ʸζ, ∂ˣθ, ∂ʸθ, ∂ˣuζ, ∂ʸvζ, ∂ˣuθ, ∂ʸvθ, 𝒟θ, 𝒟ζ, sθ, sζ) = parameters.auxiliary
    (; forcing_amplitude, ϵ, ωs) = parameters.constants

    # construct source for vorticity 
    # @. sζ = ψ
    sζ .= waver .* forcing_amplitude .* exp.(im .* φ)
    P⁻¹ * sζ

    P * θ # in place fft ζ
    P * ζ # in place fft
    # grab stream function from vorticity
    @. ψ = Δ⁻¹ * ζ
    # ∇ᵖψ
    @. u = (∂y * ψ)
    @. v = -1.0 * (∂x * ψ)
    # ∇ζ
    @. ∂ˣθ = ∂x * θ
    @. ∂ʸθ = ∂y * θ
    @. ∂ˣζ = ∂x * ζ
    @. ∂ʸζ = ∂y * ζ
    # Dissipation
    @. 𝒟ζ = 𝒟ν * ζ
    @. 𝒟θ = 𝒟κ * θ
    # go back to real space 
    P⁻¹ * u
    P⁻¹ * v
    P⁻¹ * ζ
    P⁻¹ * ∂ˣζ
    P⁻¹ * ∂ʸζ
    P⁻¹ * 𝒟ζ
    P⁻¹ * θ
    P⁻¹ * ∂ˣθ
    P⁻¹ * ∂ʸθ
    P⁻¹ * 𝒟θ
    # construct conservative form 
    @. uζ = u * ζ
    @. vζ = v * ζ
    @. uθ = u * θ
    @. vθ = v * θ
    # in place fft 
    P * uζ
    P * vζ
    P * uθ
    P * vθ
    # ∇⋅(u⃗ζ)
    @. ∂ˣuζ = ∂x * uζ
    @. ∂ʸvζ = ∂y * vζ
    # ∇⋅(u⃗θ)
    @. ∂ˣuθ = ∂x * uθ
    @. ∂ʸvθ = ∂y * vθ
    # in place ifft 
    P⁻¹ * ∂ˣuζ
    P⁻¹ * ∂ʸvζ
    P⁻¹ * ∂ˣuθ
    P⁻¹ * ∂ʸvθ

    # rhs
    @. ζ̇ = real((-u * ∂ˣζ - v * ∂ʸζ - ∂ˣuζ - ∂ʸvζ) * 0.5 + 𝒟ζ + sζ)
    @. θ̇ = real((-u * ∂ˣθ - v * ∂ʸθ - ∂ˣuθ - ∂ʸvθ) * 0.5 + 𝒟θ + sθ)
    for ω in ωs
        @. θ̇ += real(u * ϵ * cos(ω * t[1]))
    end
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

function step_perturbation!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, t, parameters)
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
    compute_perturbation_source!(parameters)
    return nothing
end

function compute_perturbation_source!(parameters) 
    (; ψ, x, y, φ, u, v, uζ, vζ, uθ, vθ, ∂ˣζ, ∂ʸζ, ∂ˣθ, ∂ʸθ, ∂ˣuζ, ∂ʸvζ, ∂ˣuθ, ∂ʸvθ, 𝒟θ, 𝒟ζ, sθ, sζ, s¹) = parameters.auxiliary
    flux_div = mean(u .* ∂ˣθ + v .* ∂ʸθ + ∂ˣuθ + ∂ʸvθ, dims=3)/2 # .- mean(u, dims=3) .* mean(∂ˣθ, dims = 3) .- mean(v, dims=3) .* mean(∂ʸθ, dims =3)
    sθ .= real.(flux_div .- u .* s¹) # flux_div # .+ 
end

function step_filter!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, t, parameters)
    rhs!(k₁, S, t, parameters)
    @. S̃ = S + Δt * k₁ * 0.5
    momentum_filter(S̃)
    momentum_filter(k₁)
    randn!(rng, φ̇)
    t[1] += Δt / 2
    @. φ += phase_speed * sqrt(Δt / 2 * 2) * φ̇ # now at t = 0.5, note the factor of two has been accounted for
    rhs!(k₂, S̃, t, parameters)
    @. S̃ = S + Δt * k₂ * 0.5
    momentum_filter(S̃)
    momentum_filter(k₂)
    rhs!(k₃, S̃, t, parameters)
    @. S̃ = S + Δt * k₃
    momentum_filter(S̃)
    momentum_filter(k₃)
    randn!(rng, φ̇)
    t[1] += Δt / 2
    @. φ += phase_speed * sqrt(Δt / 2 * 2) * φ̇ # now at t = 1.0, note the factor of two has been accounted for
    rhs!(k₄, S̃, t, parameters)
    momentum_filter(k₄)
    @. S += Δt / 6 * (k₁ + 2 * k₂ + 2 * k₃ + k₄)
    momentum_filter(S)
    return nothing
end

@info "Done with timestepping"

@info "Defining rhs timestepping for shallow water equations"
function rhs_shallow_water_conservative!(Ṡ, S, t, parameters)
    dhdt = view(Ṡ, :, :, 1)
    dhudt = view(Ṡ, :, :, 2)
    dhθdt = view(Ṡ, :, :, 3)
    h = view(S, :, :, 1)
    hu = view(S, :, :, 2)
    hθ = view(S, :, :, 3)

    (; P, P⁻¹, 𝒟ν, 𝒟κ, ∂x) = parameters.operators
    (; φ, ∂ˣhu, 𝒟h, ∂ˣhu², ∂ˣu, ∂ˣh, 𝒟hu, ∂ˣhuθ, ∂ˣθ, 𝒟hθ, shu, u, θ, hu², huθ, x) = parameters.auxiliary
    (; c, g) = parameters.constants

    # FFT 
    @. u = hu / h
    @. θ = hθ / h
    @. hu² = hu * u
    @. huθ = hu * θ
    P * h
    P * hu
    P * hθ
    P * u
    P * θ
    P * hu²
    P * huθ

    # Derivatives 
    @. ∂ˣhu² = ∂x * hu²
    @. ∂ˣhu = ∂x * hu
    @. ∂ˣu = ∂x * u
    @. ∂ˣh = ∂x * h
    @. ∂ˣhuθ = ∂x * huθ
    @. ∂ˣθ = ∂x * θ
    @. 𝒟h = 𝒟κ * h
    @. 𝒟hu = 𝒟ν * hu 
    @. 𝒟hθ = 𝒟κ * hθ 

    # IFFT 
    P⁻¹ * h
    P⁻¹ * hu
    P⁻¹ * hθ
    P⁻¹ * ∂ˣhu²
    P⁻¹ * ∂ˣhu
    P⁻¹ * ∂ˣu
    P⁻¹ * ∂ˣh
    P⁻¹ * ∂ˣhuθ
    P⁻¹ * ∂ˣθ
    P⁻¹ * 𝒟h
    P⁻¹ * 𝒟hu
    P⁻¹ * 𝒟hθ

    ## Source 
    @. shu = U * cos(x - c * t[1] + φ)

    # rhs
    @. dhdt = real(-∂ˣhu + 𝒟h)
    @. dhudt = real((-∂ˣhu² - hu / h * ∂ˣhu - hu * ∂ˣu) * 0.5 - g * h * ∂ˣh + h * shu + 𝒟hu)
    @. dhθdt = real((-∂ˣhuθ - hθ / h * ∂ˣhu - hu * ∂ˣθ) * 0.5 + 𝒟hθ)

    @. S = real(S)
    @. Ṡ = real(Ṡ)

    return nothing
end


function rhs_shallow_water!(Ṡ, S, t, parameters)
    dhdt = view(Ṡ, :, :, 1)
    dudt = view(Ṡ, :, :, 2)
    dθdt = view(Ṡ, :, :, 3)
    h = view(S, :, :, 1)
    u = view(S, :, :, 2)
    θ = view(S, :, :, 3)

    (; P, P⁻¹, 𝒟ν, 𝒟κ, 𝒟κtr, ∂x) = parameters.operators
    (; φ, ∂ˣhu, 𝒟h, ∂ˣu², ∂ˣu, ∂ˣh, 𝒟u, ∂ˣuθ, ∂ˣθ, 𝒟θ, shu, u, θ, u², uθ, x) = parameters.auxiliary
    (; U, c, g) = parameters.constants

    # FFT 
    @. hu = h * u
    @. u² = u * u
    @. uθ = u * θ
    P * h
    P * u
    P * hu

    P * θ
    P * u²
    P * uθ

    # Derivatives 
    @. ∂ˣu² = ∂x * u²
    @. ∂ˣu = ∂x * u
    @. ∂ˣhu = ∂x * hu
    @. ∂ˣh = ∂x * h
    @. ∂ˣuθ = ∂x * uθ
    @. ∂ˣθ = ∂x * θ
    @. 𝒟h = 𝒟κ * h
    @. 𝒟u = 𝒟ν * u
    @. 𝒟θ = 𝒟κtr * θ

    # IFFT 
    P⁻¹ * h
    P⁻¹ * u
    P⁻¹ * hu
    P⁻¹ * θ
    P⁻¹ * u²
    P⁻¹ * uθ

    P⁻¹ * ∂ˣu²
    P⁻¹ * ∂ˣu
    P⁻¹ * ∂ˣhu
    P⁻¹ * ∂ˣh
    P⁻¹ * ∂ˣuθ
    P⁻¹ * ∂ˣθ
    P⁻¹ * 𝒟h
    P⁻¹ * 𝒟u
    P⁻¹ * 𝒟θ

    ## Source 
    @. shu = U * cos(x - c * t[1] + φ) 

    # rhs
    @. dhdt = real(-∂ˣhu + 𝒟h)
    @. dudt = real((-∂ˣu² * 0.5 - g * ∂ˣh)  + shu + 𝒟u)
    @. dθdt = real(-∂ˣuθ + 𝒟θ)

    @. S = real(S)
    @. Ṡ = real(Ṡ)

    return nothing
end

function step_shallow_water!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, t, parameters)
    (; φ_speed, U, c, g) = parameters.constants
    rhs_shallow_water!(k₁, S, t, parameters)
    @. S̃ = S + Δt * k₁ * 0.5
    randn!(rng, φ̇)
    t[1] += Δt / 2
    @. φ += φ_speed * sqrt(Δt / 2 * 2) * φ̇ # now at t = 0.5, note the factor of two has been accounted for
    rhs_shallow_water!(k₂, S̃, t, parameters)
    @. S̃ = S + Δt * k₂ * 0.5
    rhs_shallow_water!(k₃, S̃, t, parameters)
    @. S̃ = S + Δt * k₃
    randn!(rng, φ̇)
    t[1] += Δt / 2
    @. φ += φ_speed * sqrt(Δt / 2 * 2) * φ̇ # now at t = 1.0, note the factor of two has been accounted for
    rhs_shallow_water!(k₄, S̃, t, parameters)
    @. S += Δt / 6 * (k₁ + 2 * k₂ + 2 * k₃ + k₄)
    return nothing
end

@info "Done with timestepping for shallow water equations"
