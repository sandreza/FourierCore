using HDF5, LinearAlgebra, FFTW
using GLMakie

difflist = ComplexF64[]
indmaxlist = Int[]
omegalist = Float64[]

for T in [10000.0, 25.0, 20.0, 15.0, 10.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.4, 0.3, 0.2, 0.1]
    hfile1 = h5open("time_dependent_ω_" * string(2π / T) * "_ensemble_1000.hdf5", "r")
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
    kernel_entry = -fft(ϕ̂) ./ (0.5 .* im .* fft(μ̂))
    if indmax == 1
        indmax = 1
    else
        indmax = 2
    end
    push!(indmaxlist, indmax)
    push!(difflist, kernel_entry[indmax])

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


fig2 = Figure()
phases = [angle(real(difflist[i]) + im * abs(imag(difflist[i]))) for i in 1:length(difflist)]
ax11 = Axis(fig2[1, 1], xlabel="ω", ylabel= "⟨u'c'⟩ / ⟨∂ˣc⟩")
lines!(ax11, omegalist, abs.(difflist), label="magnitude", color = :blue)
scatter!(ax11, omegalist, abs.(difflist), label="magnitude", color=:blue)
lines!(ax11, omegalist, phases ./ 2π, label="|phase| / 2π", color = :green)
scatter!(ax11, omegalist, phases ./ 2π, label="|phase| / 2π", color=:green)
axislegend(ax11, merge=true, unique=false)
display(fig2)