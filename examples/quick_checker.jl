us = []
vs = []
ψs = []
for ii in ProgressBar(1:10000)
    φ = arraytype(2π * rand(size(A)...)) # .* 0 .+ π/2 # .* 0 .+ π/3

    event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
    wait(event)
    #=
    tmpish = similar(ψ) .* 0
    for i in eachindex(𝓀ˣ), j in eachindex(𝓀ʸ)
        global tmpish .+= @. A[i,j] * cos(𝓀ˣ[i] * x + 𝓀ʸ[j] * y + φ[i,j])
    end
    =#

    P * ψ # in place fft
    # ∇ᵖψ
    @. u = -1.0 * (∂y * ψ)
    @. v = (∂x * ψ)
    # go back to real space 
    P⁻¹ * ψ
    P⁻¹ * θ
    P⁻¹ * u
    P⁻¹ * v
    u₀ = sqrt(real(mean(u .* u)))
    v₀ = sqrt(real(mean(v .* v)))
    # println(u₀)
    # println(v₀)
    # println(sqrt(mean(real.(ψ .* ψ))))
    # println(sqrt(mean(real.(tmpish .* tmpish))))
    push!(us, u₀)
    push!(vs, v₀)
    push!(ψs, sqrt(mean(real.(ψ .* ψ))))
end

#=
wavemax = 3
𝓀 = arraytype(collect(-wavemax:0.5:wavemax))
𝓀ˣ = reshape(𝓀, (length(𝓀), 1))
𝓀ʸ = reshape(𝓀, (1, length(𝓀)))
A = @. (𝓀ˣ * 𝓀ˣ + 𝓀ʸ * 𝓀ʸ)^(-1)
A[A.==Inf] .= 0.0
φ = arraytype(2π * rand(size(A)...)) .* 0 # .+ π/2 # .* 0 .+ π/3

event = stream_function!(ψ, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
wait(event)
#=
tmpish = similar(ψ) .* 0
for i in eachindex(𝓀ˣ), j in eachindex(𝓀ʸ)
    global tmpish .+= @. A[i,j] * cos(𝓀ˣ[i] * x + 𝓀ʸ[j] * y + φ[i,j])
end
=#

P * ψ # in place fft
# ∇ᵖψ
@. u = -1.0 * (∂y * ψ)
@. v = (∂x * ψ)
# go back to real space 
P⁻¹ * ψ
P⁻¹ * θ
P⁻¹ * u
P⁻¹ * v
u₀ = sqrt(real(mean(u .* u)))
v₀ = sqrt(real(mean(v .* v)))
ψ₀ = sqrt(mean(real.(ψ .* ψ)))
println(u₀ /sqrt(2))
println(mean(us))
println(v₀ /sqrt(2))
println(mean(vs))
println(ψ₀ /sqrt(2))
println(mean(ψs))
=#