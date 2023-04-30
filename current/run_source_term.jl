
f_amps = [50, 150, 300, 450, 750]
νs = [sqrt(1e-5 / 2)]
ν_hs = [sqrt(1e-3), sqrt(1e-4)]

ii = 3
jj = 1
kk = 2
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
include("timestepping.jl") # independent of everything else

filename = "case_" * string(ii) * "_" * string(jj) * "_" * string(kk)
println("---------------------------------")
println("Computing case $filename with f_amp = $f_amp, ν = $(ν^2), ν_h = $(ν_h^2)")
# Initialize the fields, choose domain size
N = 2^7
N_ens = 2^7 # 2^7
Ns = (N, N, N_ens)

include("initialize_fields.jl") # allocates memory for efficiency, defines stream function vorticity etc.

# initialize constants
κ = 1e-3 # diffusivity for scalar
dissipation_power = 2
hypoviscocity_power = 2
forcing_amplitude = f_amp * (N / 2^7)^2 # due to FFT nonsense [check if this is true]
ϵ = 0.0    # large scale parameter, 0 means off, 1 means on
ωs = [0.0]    # frequency, 0 means no time dependence
Δt = 1 / 2N # timestep
t = [0.0]  # time
kmax = 30  # filter for forcing
# initalize the operators
include("initialize_operators.jl")
# initialize ensembles and gather both eulerian and lagrangian decorrelations
scaleit = 2^3
tstart = 2^5 * scaleit # start time for gathering statistics
tend = 2^6 * scaleit # simulation endtime 2^9 is 512
mod_index = 2^3 # save every other mod index
decorrelation_index = 2^11 # how many steps till we reinitialize tracer, for lagrangian decorrelation
decorrelation_index2 = 2^13 # how many steps till we reinitialize u₀, for eulerian decorrelation
constants = (; forcing_amplitude=forcing_amplitude, ϵ=ϵ, ωs=ωs)
parameters = (; auxiliary, operators, constants) # auxiliary was defined in initialize_fields.jl
@show "initializing ensembles"
include("lagrangian_eulerian_ensemble.jl")

@info "diffusivity kernel"
using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2
# using GLMakie, HDF5
using ProgressBars
rng = MersenneTwister(1234)
Random.seed!(123456789)

maxind = minimum([40, floor(Int, N[1] / 2)])
index_choices = 2:maxind

start_index = floor(Int, tstart / Δt)

# bump(x; λ=20 / N, width=π/2)  default
bump(x; λ=40 / N, width=2) = 0.5 * (tanh((x + width / 2) / λ) - tanh((x - width / 2) / λ))
r_A = Array(@. sqrt((x - 2π)^2 + (y - 2π)^2))
θ_A = [bump(r_A[i, j]) for i in 1:N, j in 1:N]
θ .= CuArray(θ_A)
P * θ # in place fft
@. 𝒟θ = 𝒟κ * θ
P⁻¹ * 𝒟θ # in place fft
sθ .= -𝒟θ


t = [0.0]
tend = 4000
iend = ceil(Int, tend / Δt)

# new realization of flow
rand!(rng, φ) # between 0, 1
φ .*= 2π # to make it a random phase

θ̄ = arraytype(zeros(ComplexF64, N, N, N_ens))
θx_avg = arraytype(zeros(ComplexF64, N, N, N_ens))
θy_avg = arraytype(zeros(ComplexF64, N, N, N_ens))
uθ_avg = arraytype(zeros(ComplexF64, N, N, N_ens))
vθ_avg = arraytype(zeros(ComplexF64, N, N, N_ens))

iter = ProgressBar(1:iend)
ke_list = Float64[]
t .= 0.0
for i = iter
    step!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, t, parameters)
    if i % mod_index == 0
        θ_min, θ_max = extrema(real.(θ))
        ζ_min, ζ_max = extrema(real.(ζ))
        set_multiline_postfix(iter, "θ_min: $θ_min \nθ_max: $θ_max \nζ_min: $ζ_min \nζ_max: $ζ_max")
    end
    if i > start_index
        θ̄ .+= Δt .* θ
        uθ_avg .+= Δt .* u .* θ
        vθ_avg .+= Δt .* v .* θ
        θx_avg .+= Δt .* ∂ˣθ
        θy_avg .+= Δt .* ∂ʸθ
    end
    if i % mod_index == 0
        push!(ke_list, real(mean(u .* u + v .* v)))
    end
end

θ̄ ./= (tend - tstart)
uθ_avg ./= (tend - tstart)
vθ_avg ./= (tend - tstart)
θx_avg ./= (tend - tstart)
θy_avg ./= (tend - tstart)
θ̄_A = Array(real.(θ̄))
uθ_avg_A = Array(real.(uθ_avg))
vθ_avg_A = Array(real.(vθ_avg))
θx_avg_A = Array(real.(θx_avg))
θy_avg_A = Array(real.(θy_avg))

filename = "source_term_calculation"
directory = "/storage5/NonlocalPassiveTracers/Current/"

fid = h5open(directory * filename * ".hdf5", "w")
fid["θ"] = mean(θ̄_A, dims=3)[:, :, 1]
fid["uθ"] = mean(uθ_avg_A, dims=3)[:, :, 1]
fid["vθ"] = mean(vθ_avg_A, dims=3)[:, :, 1]
fid["θx"] = mean(θx_avg_A, dims=3)[:, :, 1]
fid["θy"] = mean(θy_avg_A, dims=3)[:, :, 1]
fid["source"] = Array(real.(sθ))[:,:,1]
fid["x"] = Array(x)[:]
fid["y"] = Array(y)[:]
fid["averaging start time"] = tstart
fid["averaging end time"] = tend
fid["number of ensembles"] = N_ens
close(fid)


##
using GLMakie
fig = Figure()
ax11 = Axis(fig[1, 1]; title="uθ")
ax12 = Axis(fig[1, 2]; title="vθ")
ax21 = Axis(fig[2, 1]; title="θx")
ax22 = Axis(fig[2, 2]; title="θy")
ax31 = Axis(fig[3, 1]; title="θ̄")
ax32 = Axis(fig[3, 2]; title="source")
ax41 = Axis(fig[4, 1]; title="θx vs uθ")
ax42 = Axis(fig[4, 2]; title="θy vs vθ")
lines!(ax11, sum(uθ_avg_A, dims=3)[:, 64, 1])
lines!(ax12, sum(vθ_avg_A, dims=3)[64, :, 1])
lines!(ax21, sum(θx_avg_A, dims=3)[:, 64, 1])
lines!(ax22, sum(θy_avg_A, dims=3)[64, :, 1])
lines!(ax31, sum(θ̄_A, dims=3)[:, 64, 1])
lines!(ax32, Array(real.(sum(sθ, dims=3)))[64, :, 1])
scatter!(ax41, sum(θx_avg_A, dims=3)[:, 64, 1], sum(uθ_avg_A, dims=3)[:, 64, 1])
scatter!(ax42, sum(θy_avg_A, dims=3)[64, :, 1], sum(vθ_avg_A, dims=3)[64, :, 1])
display(fig)

# scatter(sum(θx_avg_A, dims=3)[:, 64, 1], sum(uθ_avg_A, dims=3)[:, 64, 1])