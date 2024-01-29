using GLMakie


fig = Figure()
ax = Axis(fig[1,1])
# x,y=meshgrid(arange(0.5,2048),arange(0.5,1024))
x = collect(0.5:2048)
y = collect(0.5:1024)
x = reshape(x, (length(x), 1))
y = reshape(y, (1, length(y)))
ϕ = @. cos(8 * π * x / 2048) * cos(π * y/1024)
heatmap!(ax, x[:], y[:], ϕ)
display(fig)

save("/tmp/esg.jpg", fig)

##
A = randn(10,10)
