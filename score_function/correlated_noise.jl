using GLMakie, JLD2, FFTW, Statistics, ProgressBars
using FourierCore, FourierCore.Grid, FourierCore.Domain, Printf, LinearAlgebra
include("timestepping_utils.jl")
jld_name = "tracer_"
L = 2π
Ω = S¹(L)^2 × S¹(1)
N = 2^5  # number of gridpoints
Ne = 2^4 # number of ensemble members
grid = FourierGrid((N, N, Ne), Ω, arraytype = Array)
nodes, wavenumbers = grid.nodes, grid.wavenumbers

kx, ky, kz = wavenumbers
∂x = im * kx
∂y = im * ky
Δ = @. ∂x^2 + ∂y^2
##
# real.(0.01 ./ (1 .- Δ))# 
# real.(0.00001 ./ (1 .- Δ)) 
# exp.(-0.5 * sqrt.(kx .^ 2 .+ ky .^ 2)) * 1e-5 # 
Σ = real.(0.004 ./ (1 .- Δ)) # exp.(-0.5 * sqrt.(kx .^ 2 .+ ky .^ 2))   / 10^2 # 1e-6 * real.(1 .- Δ)# 
Σhalf = sqrt.(Σ)
minimum(Σ) / maximum(Σ)
fig = Figure() 
M = 5
for i in 1:M^2
    ii = (i-1) % M + 1
    jj = (i-1) ÷ M + 1
    ax = Axis(fig[ii, jj])
    correlated_noise = real.(ifft(N^2 * Σhalf .* (randn(N, N) .+ im * randn(N, N)))) 
    println(extrema(correlated_noise[:]))
    heatmap!(ax, correlated_noise[:, :, 1], colorrange = (-1.0,1.0), colormap = :balance)
end
display(fig)
##
θ = zeros(N, N, Ne) * im + randn(N, N, Ne)
P = plan_fft(θ, (1, 2))
P⁻¹ = plan_ifft(θ, (1, 2))
θ = real.(ifft(Σ .* fft(θ)))
θ ./= maximum(real.(θ))
rk = RungeKutta4(θ) 
##
θ .= zeros(N, N, Ne) * im + randn(N, N, Ne)
θ .= real.(ifft(Σ .* fft(θ)))
θ ./= maximum(real.(θ))
NN = N 
dt = 1.0/NN
κ = 1e-3/2 # * 0.1
λ = 2e3 # 4e2# 4e2
γ = 1.0^1 # 0.125^2
U = 0.02 * 2# 0.002 # 0.5
Σ⁻¹ = 1 ./ Σ
θs = []
start_index = 1000 * NN#  1000 * NN# 100 * NN# 1000 * NN
iterations = 2000 * NN # 2000 * NN# 1600 * NN # 1800 * NN
pixelval = zeros(iterations, Ne)
pixelval2 = zeros(N, N, Ne, iterations - start_index)
D = @. 1.0 * (real.(-Δ)) .^(1) # + 0.01/(real.(0.1 .- Δ))
function nonlinear_func(θ)
    # θ̂ = (P * θ)
    # ∇θ² = ((P⁻¹ * (∂x .* θ̂ ) ).^2 + (P⁻¹ * (∂y .* θ̂ ) ).^2 )
    # fθ = tanh.(θ )
    # dfθ = sech.(θ ) .^2
    # Δθ = P⁻¹ * (  Δ  .* (P * fθ)) #100 ./ (Δ .- 1) 
    # (θ² - 1) * Σ⁻¹ (θ² - 1)
    # - λ *( (sign.(θ) .+ 0.9)  - 0.1 * (dfθ) .* Δθ)
    #  P⁻¹ * (  Δ   .* (P * (-λ .* (Δθ .- 1) .* (Δθ .+ 0) .* (Δθ .+ 1))))  +
    nonlinearity = @. - λ * (θ - 1) * (θ + 0.00) * (θ + 1) # + λ * 0.01/2 * (dfθ) * Δθ  # - λ * 0.1* fθ  # - λ *( (sign.(θ) .+ 0.0) )# @. -λ  * θ * (θ^2 - γ) + λ * 0.1 * (dfθ) * Δθ  
    return nonlinearity
end

function rhs(θ)
    nonlinearity = nonlinear_func(θ)
    # extra_term = 30.0 * real.(1  .* (P⁻¹ * (  1 ./ (-1 .+ 0.5 * Δ)   .* (P * (θ )))))
    return real.(P⁻¹ * (Σ .* (P  * nonlinearity) +  (- κ * D .- U * ∂x ) .* (P * (θ))) )
end

for i in ProgressBar(1:iterations)
    ## (- ifft(Σ .* fft(θ)) ) * dt + ifft(∂x .* fft(θ)) * dt + 
    # N^2 comes from fft of white. Could take fft then divide by two (due to imaginary part variance)
    rk(rhs, θ, dt)
    noise = real.(sqrt(dt) * ( P⁻¹ * (N^2 * Σhalf .* (randn(N, N, Ne) .+ im * randn(N, N, Ne)))))
    θ .= rk.xⁿ⁺¹ + noise 
    pixelval[i, :] .= real.(θ[1, 1, :])  
    if i > start_index
        pixelval2[:, :, :, i - start_index] .= real.(θ) 
    end
    if any(isnan.(θ))
        println("NaN at iteration $i")
        break
    end
end

println("advective index timescale = ", L / U  * dt)
##
fig2 = Figure()
ax = Axis(fig2[1,1])
hist!(ax, pixelval2[:], bins = 100)
ax = Axis(fig2[1, 2])
colors = [:red, :blue, :green, :purple]
for i in 1:2
    lines!(ax, pixelval2[1, 1, i, :], color = (colors[i], 0.1))
end
ylims!(ax, (-2, 2))
ax = Axis(fig2[2, 1])
heatmap!(ax, pixelval2[:, :, 1, end], colorrange = (-2, 2), colormap = :balance)
ax = Axis(fig2[2, 2])
heatmap!(ax, pixelval2[:, :, 2, end - NN], colorrange = (-2, 2), colormap = :balance)
display(fig2)

##
Σ⁻¹ = 1 ./ Σ
function score_function(θ)
    nonlinearity = nonlinear_func(θ)
    return  real.(nonlinearity + P⁻¹ * (Σ⁻¹ .* (-κ .- κ * D) .* (P * θ)))
end
function linear_score_function(θ)
    return real.(P⁻¹ * (Σ⁻¹ .* (- κ * D) .* (P * θ)))
end
norm(score_function(θ) - linear_score_function(θ)) / norm(score_function(θ))
norm(score_function(θ) - nonlinear_func(θ)) / norm(score_function(θ))

mean(abs.(nonlinear_func(θ) ./ linear_score_function(θ)))
## exact potential is ∫dV(-θ^2/2 + θ^4/4 + θ Σ⁻¹/2 * (κ - κ * Δ) θ), exact score is θ*γ .- (θ .^3) + Σ⁻¹ * (κ - κ * Δ) θ
fig = Figure()
M = 2
# , colorrange = (-2 * sqrt(γ), 2.0 * sqrt(γ))
for i in 1:M^2
    ii = (i-1) % M + 1
    jj = (i-1) ÷ M + 1
    ax = Axis(fig[ii, jj]; title = string(i))
    heatmap!(ax, pixelval2[:, :, i, end], colormap = :balance, interpolate = true, colorrange = (-2, 2))
end
display(fig)

##
pixelval3 = pixelval2 .- mean(pixelval2);
skip = 8 * 4 * 1
endindex = size(pixelval3[:, :, :, 1:skip:end])[end]÷2
ts = skip * dt .* collect(0:endindex-1)
spatio_temporal_autocov = zeros(N, N, endindex);
for i in ProgressBar(1:Ne)
    spatio_temporal_autocov .+= real.(ifft(fft(pixelval3[:, :, i, 1:skip:end]) .* ifft(pixelval3[:, :, i, 1:skip:end]))[:, :, 1:endindex]/Ne)
end
##
covmat = zeros(N^2, N^2, endindex);
for k in ProgressBar(1:endindex)
    for i in 1:N^2
        ii = (i-1) % N + 1
        jj = (i-1) ÷ N + 1
        @inbounds covmat[:, i, k] .= circshift(spatio_temporal_autocov[:, :, k], (ii-1, jj-1))[:]
    end
end
##
C⁻¹ = inv(covmat[:, :, 1])
##
response_function = [covmat[:, :, i] * C⁻¹ for i in 1:endindex]
##
fig = Figure()
MM = 5
for i in 1:MM^2
    ii = (i-1) % MM + 1
    jj = (i-1) ÷ MM + 1
    ax = Axis(fig[ii, jj]; title = string(i))
    heatmap!(ax, nodes[1][:], nodes[2][:], circshift(abs.(reshape(response_function[5 * (i-1) + 1][:, 1], (N, N))), (N÷2, N÷2)), colormap = :afmhot, colorrange = (0, 0.1))
end
display(fig)
##
fig = Figure()
MM = 5
for i in 1:MM^2
    ii = (i-1) % MM + 1
    jj = (i-1) ÷ MM + 1
    ax = Axis(fig[ii, jj]; title = string(i))
    scatter!(ax,ts, [reshape(response_function[k][:, 1], (N, N))[mod(1 - i, N) + 1, 1] for k in 1:endindex])
end
display(fig)
##
score_response = zeros(N, N, endindex); # interaction of pixel 1 with rest
score_values = zeros(N, N, Ne, length(1:skip:size(pixelval2)[end]));
for (ii, i) in ProgressBar(enumerate(1:skip:size(pixelval2)[end]))
    score_values[:, :, :, ii] .= score_function(pixelval2[:, :, :, i])
end
for i in ProgressBar(1:Ne)   
    score_response .+= -real.(ifft(fft(pixelval2[:, :, i, 1:skip:end]) .* ifft(score_values[:, :, i, :]))[:, :, 1:endindex]/Ne/ (N^2/2))
end
##
fig = Figure()
MM = 3
for i in 1:MM^2
    ii = (i-1) % MM + 1
    jj = (i-1) ÷ MM + 1
    ax = Axis(fig[ii, jj]; title = string(ts[5 * (i-1) + 1]))
    heatmap!(ax, nodes[1][:], nodes[2][:], circshift(score_response[:, :, 5 * (i-1) + 1], (N÷2, N÷2)), colormap = :afmhot, colorrange = (0, 0.1))
end
display(fig)

##

fig = Figure()
MM = 5
for i in 1:MM^2
    ii = (i-1) % MM + 1
    jj = (i-1) ÷ MM + 1
    ax = Axis(fig[ii, jj]; title = string(i))
    modindex = i#  mod(1 - i, N) + 1
    scatter!(ax, ts, [reshape(response_function[k][:, 1], (N, N))[modindex , 1] for k in 1:endindex])
    lines!(ax, ts, score_response[modindex, 1, :], color = :purple)
    xlims!(ax, (0, 100))
end
display(fig)

##
# hack 
hack_response = zeros(N^2, N^2, endindex);
for k in ProgressBar(1:endindex)
    for i in 1:N^2
        ii = (i-1) % N + 1
        jj = (i-1) ÷ N + 1
        @inbounds hack_response[:, i, k] .= circshift(score_response[:, :, k], (ii-1, jj-1))[:]
    end
end
##
C⁻¹ = inv(hack_response[:, :, 1])
##
# hack_score_response = [(hack_response[:, :, i] * C⁻¹ + C⁻¹ * hack_response[:, :, i])/2  for i in 1:endindex]
hack_score_response = [hack_response[:, :, i] * C⁻¹ for i in 1:endindex]
##
fig = Figure()
MM = 4
for i in 1:MM^2
    ii = (i-1) % MM + 1
    jj = (i-1) ÷ MM + 1
    indexchoice = mod(8 - i, N) + 1
    ax = Axis(fig[ii, jj]; title = "1 -> " * string(indexchoice), xlabel = "time", ylabel = "response")
    lines!(ax, ts, [reshape(response_function[k][:, 1], (N, N))[indexchoice, 1] for k in 1:endindex], color = (:blue, 0.4))
    lines!(ax, ts, [reshape(hack_score_response[k][:, 1], (N, N))[indexchoice, 1] for k in 1:endindex], color = (:red, 0.4))
    xlims!(ax, (0, 50))
end
display(fig)

##
##
#=
autocor = zeros(10*N)
tmp = mean(pixelval[1:end,:] .* pixelval[1:end, :]) 
tmp2 =  mean(pixelval)^2
for j in 1:10 * N
    autocor[j] = (mean(pixelval[1:end-j+1, :] .* pixelval[j:end, :])) / tmp 
end
fig = Figure()
ax = Axis(fig[1,1])
scatter!(ax, autocor)
ylims!(ax, (-0.5, 1.0))
display(fig)
##
endindex = 10*N
autocor = zeros(endindex)
tmp = mean(pixelval2[:, :, :, 1:end] .* pixelval2[:, :, :, 1:end]) 
tmp2 =  mean(pixelval2)^2
for j in ProgressBar(1:endindex)
    autocor[j] = (mean(pixelval2[:, :, :, 1:end-j+1] .* pixelval2[:, :, :, j:end])) / tmp 
end
fig = Figure()
ax = Axis(fig[1,1])
scatter!(ax, autocor)
ylims!(ax, (-0.5, 1.1))
display(fig)
=#
##
#=
endindex = 100*N
autocor = zeros(endindex, Ne)
tmp = mean(pixelval2 .* pixelval2, dims = (1, 2, 4)) 
# tmp2 =  @. mean(pixelval2, )^2
for j in ProgressBar(1:endindex)
    autocor[j, :] .= ((mean(pixelval2[:, :, :, 1:end-j+1] .* pixelval2[:, :, :, j:end], dims = (1, 2, 4))) ./ tmp )[:]
end
=#
##
#=
ts = dt .* collect(0:endindex-1)
fig = Figure()
ax = Axis(fig[1,1])
for i in 1:Ne
    lines!(ax, ts, autocor[:, i], color = (:blue, 0.1))
end
lines!(ax, ts, mean(autocor, dims = 2)[:], color = (:black, 1.0))
ts2 = L / U .* collect(0:10)
vlines!(ax, ts2, color = :red )
=#
##
#=
xlims!(ax, (0, 100))
ylims!(ax, (-0.5, 1.1))
display(fig)
=#
##

##
γ = 10.0
xvals = range(-3, 3, length = 100)
scatter(log.(cosh.(γ * xvals)))