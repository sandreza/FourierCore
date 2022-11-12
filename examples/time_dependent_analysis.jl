using HDF5, LinearAlgebra, FFTW
using GLMakie

difflist = ComplexF64[]
indmaxlist = Int[]
omegalist = Float64[]
# hfile1 = h5open("time_dependent_ω_0.0006283185307179586_ensemble_1000.hdf5", "r")
for T in [10000.0, 25.0, 20.0, 15.0, 10.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.4, 0.3, 0.2, 0.1]
    hfile1 = h5open("time_dependent_ω_" * string(2π / T) * "_ensemble_1000.hdf5", "r")
    # hfile1 = h5open("time_dependent_ω_0.25132741228718347_ensemble_1000.hdf5", "r")
    t = read(hfile1["time"])
    μ = read(hfile1["ensemble mean"])
    ϕ = read(hfile1["ensemble flux"])
    ω = read(hfile1["omega"])
    push!(omegalist, ω)
    T = 2π / ω
    Δt = (t[2] - t[1])
    m = round(Int, T / Δt)
    if m > length(t)
        m = 2000
    end
    ϕ̂ = fft(ϕ, 1)[2, end-m+1:end]
    μ̂ = fft(μ, 1)[2, end-m+1:end]
    indmax = argmax(abs.(fft(ϕ̂)[:]))
    totes = -fft(ϕ̂) ./ (0.5 .* im .* fft(μ̂))
    push!(indmaxlist, indmax)
    push!(difflist, totes[indmax])

    close(hfile1)
end

fig = Figure()
ax11 = Axis(fig[1, 1], xlabel="ω", ylabel= "⟨u'c'⟩ / ⟨∂ˣc⟩")
lines!(ax11, omegalist, real.(difflist), label="|real|", color = :blue)
scatter!(ax11, omegalist, real.(difflist), label="|real|", color=:blue)
lines!(ax11, omegalist, abs.(imag.(difflist)), label="|imaginary|", color = :green)
scatter!(ax11, omegalist, abs.(imag.(difflist)), label="|imaginary|", color=:green)
axislegend(ax11, merge=true, unique=false)
display(fig)