include("initialize_shallow_water_fields.jl")
Random.seed!(12345)
filename = "shallow water"
directory = "/storage5/NonlocalPassiveTracers/Current/"
fid = h5open(directory * filename * ".hdf5", "w")
##
φ_speeds = collect(0:0.1:1)
κtrs = [1e-2, 5e-2, 0.15, 0.2]
case_number = 0
for κtr in ProgressBar(κtrs)
for φ_speed in ProgressBar(φ_speeds)

global case_number += 1
# φ_speed = 0.3
# κtr = 5e-2

c = 0.1
g = 1.0
U = 1.0

rand!(rng, φ)
φ .*= 2π


ν = 0.2 # 0.1 * Δx^2 / Δt
κ = 0.2 # 0.1 * Δx^2 / Δt

Δx = x[2] - x[1]
cfl = 0.2
Δt = cfl * Δx / maximum([U, c, κ / Δx, ν / Δx])


𝒟ν = @. ν * Δ
𝒟κ = @. κ * Δ
𝒟κtr = @. κtr * Δ

operators = (; P, P⁻¹, 𝒟ν, 𝒟κ, 𝒟κtr, ∂x)
constants = (; φ_speed, U, c, g)
auxiliary = (; φ, ∂ˣhu, 𝒟h, ∂ˣu², ∂ˣu, ∂ˣh, 𝒟u, ∂ˣuθ, ∂ˣθ, 𝒟θ, shu, u, θ, u², uθ, x)
parameters = (; operators, constants, auxiliary)
t = [0.0]

##
Tend = 200
iterations = floor(Int, Tend / Δt)
timesnapshots_u = Vector{Float64}[]
timesnapshots_h = Vector{Float64}[]
timesnapshots_θ = Vector{Float64}[]
ensemble_mean_flux = Vector{Float64}[]
ensemble_mean_flux_h = Vector{Float64}[]
for i in ProgressBar(1:iterations)
    step_shallow_water!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, t, parameters)
    if i % 10 == 0
        push!(timesnapshots_u, Array(real.(u)[:, 1]))
        push!(timesnapshots_h, Array(real.(h)[:, 1]))
        push!(timesnapshots_θ, Array(real.(θ)[:, 1]))
        push!(ensemble_mean_flux, Array(mean(real(u .* θ), dims=2)[:]))
        push!(ensemble_mean_flux_h, Array(mean(real(u .* h), dims=2)[:]))
    end
end
x_A = Array(x)
##
#=
fig = Figure()
ax11 = Axis(fig[1, 1]; title="h")
ax21 = Axis(fig[2, 1]; title="u")
ax31 = Axis(fig[3, 1]; title="θ")
ax41 = Axis(fig[4, 1]; title="uθ")
sl_x = Slider(fig[5, 1], range=1:length(timesnapshots_u), startvalue=1)
o_index = sl_x.value
field = @lift timesnapshots_h[$o_index]
field2 = @lift timesnapshots_u[$o_index]
field3 = @lift timesnapshots_θ[$o_index]
field4 = @lift ensemble_mean_flux[$o_index]

lines!(ax11, x_A[:], field)
ylims!(ax11, (0.0, 2.5))
lines!(ax21, x_A[:], field2)
ylims!(ax21, (-0.5, 0.5))
lines!(ax31, x_A[:], field3)
ylims!(ax31, (0, 2.5))
lines!(ax31, x_A[:], field3)
ylims!(ax31, (0, 2.5))
lines!(ax41, x_A[:], field4)
ylims!(ax41, (-0.0, 0.5))
display(fig)
=#
##
mean(ensemble_mean_flux[end])
##
tmparray = zeros(Ns[1])
for i in floor(Int, length(ensemble_mean_flux) / 2):length(ensemble_mean_flux)
    tmparray .+= ensemble_mean_flux[i]
end
tmparray .= tmparray / length(floor(Int, length(ensemble_mean_flux) / 2):length(ensemble_mean_flux))
flux = mean(tmparray)
tmp2 = [mean(ensemble_mean_flux[i]) for i in length(ensemble_mean_flux)-100:length(ensemble_mean_flux)]
tmp2_hist = [mean(ensemble_mean_flux[i]) for i in 1:length(ensemble_mean_flux)]
tlist = 10Δt .* collect(1:length(tmp2_hist))
scatter(tlist, tmp2_hist)

tmparray = zeros(Ns[1])
for i in floor(Int, length(ensemble_mean_flux) / 2):length(ensemble_mean_flux)
    tmparray .+= ensemble_mean_flux_h[i]
end
tmparray .= tmparray / length(floor(Int, length(ensemble_mean_flux) / 2):length(ensemble_mean_flux))
flux_h = mean(tmparray)


fid["kappa tracer $case_number"] = κtr
fid["phase speed $case_number"] = φ_speed
fid["N $case_number"] = Ns[1]
fid["Nensemble $case_number"] = Ns[2]
fid["nu $case_number"] = ν
fid["kappa $case_number"] = κ
fid["flux h $case_number"] = flux_h
fid["flux tracer $case_number"] = flux
fid["flux tracer timehistory $case_number"] = tmp2_hist 
fid["flux tracer time list $case_number"] = tlist


end
end

close(fid)