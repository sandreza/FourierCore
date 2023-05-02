# operators
∂x = im * kˣ
∂y = im * kʸ
Δ = @. ∂x^2 + ∂y^2

# plan ffts
P = plan_fft!(ψ, (1, 2))
P⁻¹ = plan_ifft!(ψ, (1, 2))

# Dissipation 
Δ = @. ∂x^2 + ∂y^2
Δ⁻¹ = 1 ./ Δ
bools = (!).(isnan.(Δ⁻¹))
Δ⁻¹ .*= bools # hack in the fact that false * NaN = 0

𝒟ν = @. -(-ν_h * Δ⁻¹)^(hypoviscocity_power) - (-ν * Δ)^(dissipation_power)
𝒟κ = @. κ * Δ

# filter for forcing 
# construct waver
kxmax = maximum(kˣ)
kymax = maximum(kʸ)
kxymax = maximum([kxmax, kymax])
waver = @. (kˣ)^2 + (kʸ)^2 ≤ 0.5 * kxymax^2
waver .*= @. (kˣ != 0.0) .* (kʸ != 0.0)
waver[1, :] .= 1.0
waver[:, 1] .= 1.0
waver[1, 1] = 0.0
#=
kxmax = maximum(kˣ)
kymax = maximum(kˣ)
kxmax = kymax = kmax
waver = @. (kˣ)^2 + (kʸ)^2 ≤ ((kxmax / 2)^2 + (kymax / 2)^2)
waver = @. abs(kˣ) .+ 0 * abs(kʸ) ≤ 2 / 3 * kxmax
@. waver = waver * (0 * abs(kˣ) .+ 1 * abs(kʸ) ≤ 2 / 3 * kxmax)
waver[1, 1] = 0.0
waver[:, floor(Int, N / 2)+1] .= 0.0
waver[floor(Int, N / 2)+1, :] .= 0.0
=#

operators = (; P, P⁻¹, Δ⁻¹, waver, 𝒟ν, 𝒟κ, ∂x, ∂y)
