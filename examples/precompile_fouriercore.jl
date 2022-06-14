using FourierCore, FourierCore.Grid, FourierCore.Domain
using FFTW, LinearAlgebra, BenchmarkTools, Random, JLD2
using GLMakie, CUDA

a = CuArray(randn(128, 128)) .+ im
P = plan_fft!(a)
Pi = plan_ifft!(a)
Pa = plan_fft!(Array(a))
Pia = plan_ifft!(Array(a))

arraytype = CuArray
Ω = S¹(2π)^2
N = 2^5 # number of gridpoints
grid = FourierGrid(N, Ω, arraytype=arraytype)


heatmap(randn(10, 10))
lines(randn(10))
contour(randn(10, 10))
volume(randn(10, 10, 10))
fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, randn(10))
display(fig)
