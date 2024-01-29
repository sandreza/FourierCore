using ProgressBars, Random, GLMakie , FFTW, Statistics
include("timestepping_utils.jl")
Random.seed!(1234)

dt = 0.01 
iterations = 10^5
function ou(x)
    # -sign.(x) + 0.0 # 
    return @.  -(x-1)*(x+0.0)*(x+1)# - (sign(x+1) * abs(x-1) + sign(x-1) * abs(x+1) + 0.1)# x * (1- x^2)
end

Ne = 1000
trajectory = randn(Ne, iterations)
trajectory[:, 1] .= zeros(Ne)
step = RungeKutta4(randn(Ne))
step2 = RungeKutta4(randn(Ne))
ϵ = 0.4
for i in ProgressBar(2:iterations)
    step(ou, trajectory[:, i-1], dt)
    trajectory[:, i] .= step.xⁿ⁺¹ .+ ϵ * sqrt(dt*2) * randn(Ne)
end
trajectory[:, 1] .= trajectory[:, end]
for i in ProgressBar(2:iterations)
    step(ou, trajectory[:, i-1], dt)
    trajectory[:, i] .= step.xⁿ⁺¹ .+ ϵ * sqrt(dt*2) * randn(Ne)
end

trajectory2 = trajectory .- mean(trajectory)
autocov = mean([real.(ifft(fft(trajectory2[i, :]) .* ifft(trajectory2[i, :])))[1:iterations÷2] for i in 1:Ne])
autocov2 = mean([-real.(ifft(fft(trajectory[i, :]) .* ifft(ou.(trajectory[i, :]))))[1:iterations÷2] / ϵ^2 for i in 1:Ne]) 
##
inds = 1:10:4000
ts = (collect(inds) .- 1) * dt
fig = Figure() 
ax = Axis(fig[1,1]; title = "response")
scatter!(ax,ts,  autocov[inds] / autocov[1], color = (:red, 0.5), label = "Gaussian")
scatter!(ax,ts,  autocov2[inds]/ autocov2[1], color = (:blue, 0.5), label = "Score")
axislegend(ax, position = :rt)
ax = Axis(fig[1,2]; title = "distribution")
hist!(ax, trajectory[:], bins = 100)
display(fig)
##
δ = 10*dt
iterations = inds[end]
trajectory2 = copy(trajectory)* 0 
trajectory2[:, 1] .= trajectory[:, 1] .+ δ
for i in ProgressBar(2:iterations)
    noise = ϵ * sqrt(dt*2) * randn(Ne)
    step(ou, trajectory[:, i-1], dt)
    trajectory[:, i] .= step.xⁿ⁺¹ + noise
    step2(ou, trajectory2[:, i-1], dt)
    trajectory2[:, i] .= step2.xⁿ⁺¹ + noise
end
Δtrajectory = mean(trajectory2[:, inds] - trajectory[:, inds], dims = 1)[:] / δ
##
ts = (collect(inds) .- 1) * dt
fig = Figure(resolution = (689, 277)) 
ax = Axis(fig[1,1]; title = "response")
scatter!(ax,ts,  autocov[inds] / autocov[1], color = (:red, 0.5), label = "Gaussian")
scatter!(ax,ts,  autocov2[inds]/ autocov2[1], color = (:blue, 0.5), label = "Score")
scatter!(ax,ts,  Δtrajectory, color = (:orange, 0.5), label = "Numerical")
axislegend(ax, position = :rt)
ax = Axis(fig[1,2]; title = "distribution")
hist!(ax, trajectory[:], bins = 100)
display(fig)