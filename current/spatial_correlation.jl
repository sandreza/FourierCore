end_ind = 2^9 * 2^3
iter = ProgressBar(1:end_ind)
ke_list = Float64[]
Ns = 128
uu_cor = zeros(Ns, Ns)
uv_cor = zeros(Ns, Ns)
vu_cor = zeros(Ns, Ns)
vv_cor = zeros(Ns, Ns)
t .= 0.0
rolling_index = 0
for i = iter
    step!(S, S̃, φ, φ̇, k₁, k₂, k₃, k₄, Δt, rng, t, parameters)
    if i%2^6 == 0
        rolling_index += 1
        for ii in 0:Ns-1, jj in 0:Ns-1
            uu_cor[ii+1, jj+1] += mean(real.(circshift(u, (ii,jj,0))) .* real(u)) 
            uv_cor[ii+1, jj+1] += mean(real.(circshift(u, (ii,jj,0))) .* real(v)) 
            vu_cor[ii+1, jj+1] += mean(real.(circshift(v, (ii,jj,0))) .* real(u)) 
            vv_cor[ii+1, jj+1] += mean(real.(circshift(v, (ii,jj,0))) .* real(v)) 
        end
    end
end
uu_cor ./= rolling_index
uv_cor ./= rolling_index
vu_cor ./= rolling_index
vv_cor ./= rolling_index
##
fig = Figure() 
ax = Axis(fig[1,1], xlabel = "x", ylabel = "uu")
lines!(ax, uu_cor[:, 1])
ax = Axis(fig[1,2], xlabel = "x", ylabel = "uv")
lines!(ax, uv_cor[:, 1])
ax = Axis(fig[2,1], xlabel = "x", ylabel = "vu")
lines!(ax, vu_cor[:, 1])
ax = Axis(fig[2,2], xlabel = "x", ylabel = "vv")
lines!(ax, vv_cor[:, 1])
display(fig)
##
fig = Figure() 
ax = Axis(fig[1,1], xlabel = "x", ylabel = "uu")
lines!(ax, [uu_cor[i,i] for i in 1:Ns])
ax = Axis(fig[1,2], xlabel = "x", ylabel = "uv")
lines!(ax, [uv_cor[i,i] for i in 1:Ns])
ax = Axis(fig[2,1], xlabel = "x", ylabel = "vu")
lines!(ax, [vu_cor[i,i] for i in 1:Ns])
ax = Axis(fig[2,2], xlabel = "x", ylabel = "vv")
lines!(ax, [vv_cor[i,i] for i in 1:Ns])
display(fig)
##
fig = Figure() 
ax = Axis(fig[1,1], xlabel = "x", ylabel = "uu")
heatmap!(ax, uu_cor, colorrange = (-0.2, 0.2), colormap = :balance)
ax = Axis(fig[1,2], xlabel = "x", ylabel = "uv")
heatmap!(ax, uv_cor, colorrange = (-0.05, 0.05), colormap = :balance)
ax = Axis(fig[2,1], xlabel = "x", ylabel = "vu")
heatmap!(ax, vu_cor, colorrange = (-0.05, 0.05), colormap = :balance)
ax = Axis(fig[2,2], xlabel = "x", ylabel = "vv")
heatmap!(ax, vv_cor, colorrange = (-0.2, 0.2), colormap = :balance)
display(fig)