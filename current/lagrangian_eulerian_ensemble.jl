sθ .= 0.0

tstart = 250.0
tend = 500.0

iend = ceil(Int, tend / Δt)
start_index = floor(Int, tstart / Δt)
eulerian_list = Float64[]
lagrangian_list = Float64[]
ke_list = Float64[]
tlist = Float64[]

mod_index = 10 # save every other mod index
decorrelation_index  = 4000 # how many steps till we reinitialize tracer
decorrelation_index2 = 12000 # how many steps till we reinitialize u₀

t = [0.0]
iter = ProgressBar(1:iend)
for i = iter
    step!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, parameters)
    t[1] += Δt

    if i == start_index
        θ .= u
        u₀ .= u
    end

    if (i > start_index) && (i % mod_index == 0)
        if i % decorrelation_index == 0
            # decorrelates after 4000 timesteps
            θ .= u
        end
        if i % decorrelation_index2 == 0
            u₀ .= u
        end
        uu = real(mean(u .* u₀))
        tmpuθ = real(mean(u .* θ))

        push!(eulerian_list, uu)
        push!(lagrangian_list, tmpuθ)
        push!(ke_list, real(mean(u .* u + v .* v)))
        push!(tlist, t[1])

        θ_min, θ_max = extrema(real.(θ))
        ζ_min, ζ_max = extrema(real.(ζ))
        set_multiline_postfix(iter, "θ_min: $θ_min \nθ_max: $θ_max \nζ_min: $ζ_min \nζ_max: $ζ_max")
    end
end
