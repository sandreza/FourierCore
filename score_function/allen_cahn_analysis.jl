include("allehn_cahn_model.jl")
##
pv, sf = allehn_cahn()
lr = linear_response_function(pv)
hsr = hack_score_response_function(pv, sf)

##
fig = Figure(resolution = (857, 700))
MM = 4
N = 32
ts = collect(0:length(hsr)-1)
for i in 1:MM^2
    ii = (i-1) % MM + 1
    jj = (i-1) ÷ MM + 1
    indexchoice = mod(8 - i, N) + 1
    ax = Axis(fig[ii, jj]; title = "1 -> " * string(indexchoice), xlabel = "time", ylabel = "response")
    lines!(ax, ts, [reshape(lr[k][:, 1], (N, N))[indexchoice, 1] for k in eachindex(lr)], color = (:blue, 0.4))
    lines!(ax, ts, [reshape(hsr[k][:, 1], (N, N))[indexchoice, 1] for k in eachindex(lr)], color = (:red, 0.4))
    xlims!(ax, (0, 50))
end
display(fig)
##
save(pwd() * "/response.png", fig)

##
#=
fig = Figure(resolution = (3*300, 3*300))
time_index = Observable(1)
fields = [@lift(pv[:, :, i, $time_index]) for i in 1:4]
for i in 1:4
    ii = (i-1)%2 + 1
    jj = (i-1)÷2 + 1
    ax = Axis(fig[ii,jj]; title = @lift("ensemble member $i at t=" * string($time_index÷32)))
    heatmap!(ax, fields[i], colorrange = (-2, 2), colormap = :balance, interpolate = true)
    display(fig)
end
framerate = 30
timestamps = 32:32:size(pv)[end]
record(fig, pwd() * "/generic.mp4", timestamps; framerate = framerate) do t
    time_index[] = t
end
=#

##
parameters_list = []
scale_list = reverse([0.5, 1, 2.0])
# , ℓ in scale_list
for i in scale_list, j in scale_list, k in scale_list
    parameters = (; N = 32, Ne = 2^2, ϵ² = 0.004, κ = 1e-3/2*i, U = 0.04 * j, λ = 2e3 * k)
    push!(parameters_list, parameters)
end
##
i = 0
for parameters in ProgressBar(parameters_list)
    pv, sf = allehn_cahn(; parameters)
    lr = linear_response_function(pv)
    hsr = hack_score_response_function(pv, sf)

    fig = Figure(resolution = (857, 700))
    MM = 4
    N = 32
    ts = collect(0:length(hsr)-1)
    for i in 1:MM^2
        ii = (i-1) % MM + 1
        jj = (i-1) ÷ MM + 1
        indexchoice = mod(8 - i, N) + 1
        ax = Axis(fig[ii, jj]; title = "1 -> " * string(indexchoice), xlabel = "time", ylabel = "response")
        lines!(ax, ts, [reshape(lr[k][:, 1], (N, N))[indexchoice, 1] for k in eachindex(lr)], color = (:blue, 0.4))
        lines!(ax, ts, [reshape(hsr[k][:, 1], (N, N))[indexchoice, 1] for k in eachindex(lr)], color = (:red, 0.4))
        xlims!(ax, (0, 100))
    end
    display(fig)  
    i += 1  
    save(pwd() * "/response_$i.png", fig)
end

##
parameters_list[22]
parameters_list[25]
p = (; N = 32, Ne = 2^5, ϵ² = 0.004, κ = 1e-3/4, U = 0.02, λ = 4e3)
pv, sf = allehn_cahn(; parameters = p)

#=
fig = Figure(resolution = (3*300, 3*300))
time_index = Observable(1)
fields = [@lift(pv[:, :, i, $time_index]) for i in 1:4]
for i in 1:4
    ii = (i-1)%2 + 1
    jj = (i-1)÷2 + 1
    ax = Axis(fig[ii,jj]; title = @lift("ensemble member $i at t=" * string($time_index÷32)))
    heatmap!(ax, fields[i], colorrange = (-2, 2), colormap = :balance, interpolate = true)
    display(fig)
end

framerate = 30
timestamps = 32:32:size(pv)[end]
record(fig, pwd() * "/generic_3.mp4", timestamps; framerate = framerate) do t
    time_index[] = t
end
=#

lr = linear_response_function(pv)
sr = score_response_function(pv, sf)
hsr = hack_score_response_function(pv, sf)
##
fig = Figure(resolution = (857, 700))
MM = 4
N = 32
ts = collect(0:length(hsr)-1)
for i in 1:MM^2
    ii = (i-1) % MM + 1
    jj = (i-1) ÷ MM + 1
    indexchoice = mod(8 - i, N) + 1
    ax = Axis(fig[ii, jj]; title = "1 -> " * string(indexchoice), xlabel = "time", ylabel = "response")
    lines!(ax, ts, [reshape(lr[k][:, 1], (N, N))[indexchoice, 1] for k in eachindex(lr)], color = (:blue, 0.4))
    lines!(ax, ts, [reshape(hsr[k][:, 1], (N, N))[indexchoice, 1] for k in eachindex(lr)], color = (:red, 0.4))
    xlims!(ax, (0, 100))
end
display(fig)    

##
fig2 = Figure()
ax = Axis(fig2[1,1])
hist!(ax, pv[:], bins = 100)
ax = Axis(fig2[1, 2])
colors = [:red, :blue, :green, :purple]
for i in 1:2
    lines!(ax, pv[1, 1, i, :], color = (colors[i], 0.1))
end
ylims!(ax, (-2, 2))
ax = Axis(fig2[2, 1])
heatmap!(ax, pv[:, :, 1, end], colorrange = (-2, 2), colormap = :balance, interpolate = true)
ax = Axis(fig2[2, 2])
heatmap!(ax, pv[:, :, 2, end ], colorrange = (-2, 2), colormap = :balance, interpolate = true)
display(fig2)
##
fig = Figure(resolution = (3*300, 3*300))
time_index = Observable(1)
fields = [@lift(pv[:, :, i, $time_index]) for i in 1:4]
for i in 1:4
    ii = (i-1)%2 + 1
    jj = (i-1)÷2 + 1
    ax = Axis(fig[ii,jj]; title = @lift("ensemble member $i at t=" * string($time_index÷32)))
    heatmap!(ax, fields[i], colorrange = (-2, 2), colormap = :balance, interpolate = true)
    display(fig)
end

framerate = 30
timestamps = 32:32:size(pv)[end]
record(fig, pwd() * "/generic_4.mp4", timestamps; framerate = framerate) do t
    time_index[] = t
end

##
fig3 = Figure(resolution = (3*300, 3*300))
time_index = Observable(1)
ax = Axis(fig3[1,1]; title = @lift("Response Function t= " *string($time_index)))
field = @lift(circshift(abs.(sr[:, :, $time_index]), (N÷2, N÷2))) 
heatmap!(ax, field, colorrange = (0, 0.2), colormap = :grays)
display(fig3)

framerate = 20
timestamps = 1:100
record(fig3, pwd() * "/response_4.mp4", timestamps; framerate = framerate) do t
    time_index[] = t
end


##
p = (; N = 32, Ne = 2^5, ϵ² = 0.004, κ = 1e-3/4, U = 0.02, λ = 4e3)
pv, sf = allehn_cahn(; parameters = p)
##
fig = Figure(resolution = (3*300, 3*300))
time_index = Observable(1)
fields = [@lift(pv[:, :, i, $time_index]) for i in 1:4]
for i in 1:4
    ii = (i-1)%2 + 1
    jj = (i-1)÷2 + 1
    ax = Axis(fig[ii,jj]; title = @lift("ensemble member $i at t=" * string($time_index÷32)))
    heatmap!(ax, fields[i], colorrange = (-2, 2), colormap = :balance, interpolate = true)
    display(fig)
end

framerate = 30
timestamps = 32:32:size(pv)[end]
record(fig, pwd() * "/generic_one_start.mp4", timestamps; framerate = framerate) do t
    time_index[] = t
end
