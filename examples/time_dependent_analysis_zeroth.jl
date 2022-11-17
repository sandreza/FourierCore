using HDF5, LinearAlgebra, FFTW, Statistics
# using GLMakie

difflist = ComplexF64[]
indmaxlist = Int[]
omegalist = Float64[]
ϕlist = []
Tlist = [10000.0, 25.0, 20.0, 15.0, 10.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.4, 0.3, 0.2, 0.1]
for T in Tlist
    hfile1 = h5open("time_dependent_ω_" * string(2π / T) * "_ensemble_1000_zeroth.hdf5", "r")
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
    ϕ̂ = mean(ϕ, dims=1)[end-m+1:end]
    μ̂ = mean(μ, dims=1)[end-m+1:end]
    forcing = sin.(ω * t[end-m+1:end])
    f̂ = fft(forcing)[2]
    indmax = argmax(abs.(fft(ϕ̂)[:]))
    if indmax > 1
        indmax = 2
        kernel_entry = (fft(ϕ̂) ./ f̂) ./ im
    else
        indmax = 1
        kernel_entry = abs.(fft(ϕ̂) / m / im)
    end
    push!(indmaxlist, indmax)
    push!(difflist, kernel_entry[indmax])
    push!(ϕlist, ϕ̂)

    close(hfile1)
end

#=
fig = Figure()
ax11 = Axis(fig[1, 1], xlabel="ω", ylabel= "⟨u'c'⟩ ")
lines!(ax11, omegalist, abs.(real.(difflist)), label="|real|", color = :blue)
scatter!(ax11, omegalist, abs.(real.(difflist)), label="|real|", color=:blue)
lines!(ax11, omegalist, abs.(imag.(difflist)), label="|imaginary|", color = :green)
scatter!(ax11, omegalist, abs.(imag.(difflist)), label="|imaginary|", color=:green)
axislegend(ax11, merge=true, unique=false)
display(fig)
=#

fig2 = Figure()
phases = -[angle(real(difflist[i]) + im * imag(difflist[i])) for i in 1:length(difflist)]
ax11 = Axis(fig2[1, 1], xlabel="ω", ylabel="magnitude", yticklabelcolor = :blue, ytickcolor = :blue, leftspinecolor = :blue, ylabelcolor = :blue)
ax12 = Axis(fig2[1, 1], xlabel="ω", ylabel="phase", yticklabelcolor= :green, ytickcolor = :green, leftspinecolor = :blue, rightspinecolor = :green, yaxisposition = :right, ylabelcolor = :green)
lines!(ax11, omegalist, abs.(difflist), label="magnitude", color=:blue)
scatter!(ax11, omegalist, abs.(difflist), label="magnitude", color=:blue)
# lines!(ax11, omegalist, phases ./ 2π, label="phase / 2π", color=:green, )
# scatter!(ax11, omegalist, phases ./ 2π, label="phase / 2π", color=:green)
lines!(ax12, omegalist, phases, label="phase", color=:green)
scatter!(ax12, omegalist, phases, label="phase", color=:green)
# axislegend(ax11, merge=true, unique=false, position = :cc)
# axislegend(ax12, merge=true, unique=false, position = :rc)
display(fig2)

