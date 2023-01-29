sθ .= 0.0

#=
scaleit = 2^3
tstart = 2^5 * scaleit
tend = 2^6 * scaleit
are defined outside
=#

iend = ceil(Int, tend / Δt)
start_index = round(Int, tstart / Δt)
eulerian_list = Float64[]
lagrangian_list = Float64[]
ke_list = Float64[]
tlist = Float64[]

#=
mod_index = 2^3 # save every other mod index
decorrelation_index = 2^11 # how many steps till we reinitialize tracer
decorrelation_index2 = 2^13 # how many steps till we reinitialize u₀
are defined outside
=#
t .= 0.0
iter = ProgressBar(1:iend)
for i = iter
    step!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, t, parameters)
    if i == start_index
        θ .= u
        u₀ .= u
    end
    if (i > start_index) && (i % mod_index == 0)
        if i % decorrelation_index == 0
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
        s1 = "θ_min: $θ_min \nθ_max: $θ_max \nζ_min: $ζ_min \nζ_max: $ζ_max"
        s2 = "\nuu : $(uu[i]) \nuθ : $(tmpuθ[i])"
        set_multiline_postfix(iter, s1 * s2)
    end
end
