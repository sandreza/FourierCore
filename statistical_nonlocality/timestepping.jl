using Random, ProgressBars

struct StochasticRungeKutta4{S1,S2,F1,F2,P,T}
    auxiliary::S1
    stochastic_auxiliary::S2
    rhs!::F1
    stochastic_rhs!::F2
    parameters::P
    timetracker::T
end

function (step!::StochasticRungeKutta4)(S, φ, rng)
    S̃, k₁, k₂, k₃, k₄ = step!.auxiliary
    Δt = step!.timetracker[2]
    t = view(step!.timetracker, 1)
    rhs! = step!.rhs!
    parameters = step!.parameters

    rhs!(k₁, S, φ, t, parameters)
    @. S̃ = S + Δt * k₁ * 0.5
    t[1] += Δt / 2
    step!(φ, rng)
    rhs!(k₂, S̃, φ, t, parameters)
    @. S̃ = S + Δt * k₂ * 0.5
    rhs!(k₃, S̃, φ, t, parameters)
    @. S̃ = S + Δt * k₃
    t[1] += Δt / 2
    step!(φ, rng)
    rhs!(k₄, S̃, φ, t, parameters)
    @. S += Δt / 6 * (k₁ + 2 * k₂ + 2 * k₃ + k₄)
end

function (stochastic_step!::StochasticRungeKutta4)(φ, rng)
    φ̇ = stochastic_auxiliary
    Δt = stochastic_step!.timetracker[2]
    rhs! = stochastic_step!.stochastic_rhs!
    parameters = step!.parameters.stochastic
    ϵ = parameters.noise_amplitude
    rhs!(φ̇, φ, parameters)
    @. φ += φ̇ * Δt/2
    randn!(rng, φ̇)
    @. φ += ϵ * sqrt(Δt/2 * 2) * φ̇ # now at t = 0.5, note the factor of two has been accounted for
end

function ou_rhs!(φ̇, φ, parameters)
    γ = parameters.ou_amplitude
    @. φ̇ = -γ * φ
end

function advection_rhs!(Ṡ, S, u, t, parameters)
    (; ∂x , 𝒟) = parameters.operators
    source = parameters.source
    @. Ṡ = -u * ∂x * S + 𝒟 * S + source
end