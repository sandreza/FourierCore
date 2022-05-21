fig = Figure()
ax = Axis(fig[1, 1])

fid = h5open(filename, "r")

stringslider = Slider(fig[2, 1], range=1:120, startvalue=0)
slidervalue = stringslider.value
field = @lift(read(fid["moisture"][string($slidervalue)]))
colorrange = @lift(extrema($field))
heatmap!(ax, field, colormap = :balance, colorrange = colorrange)