using HDF5, GLMakie


filename = "source_term_calculation_preliminary"
directory = "/storage5/NonlocalPassiveTracers/Current/"

fid = h5open(directory * filename * ".hdf5", "r")
∂ˣθ = read(fid["θx"])
∂ʸθ = read(fid["θy"])
x = read(fid["x"])
y = read(fid["y"])
uθ = read(fid["uθ"])
vθ = read(fid["vθ"])
close(fid)

x = reshape(x, (128, 1)) .- 2π
y = reshape(y, (1, 128)) .- 2π

r = @. (x^2 + y^2)^(1 / 2)
ĉ = @. x / r
ŝ = @. y / r
ĉ[isnan.(ĉ)] .= 0
ŝ[isnan.(ŝ)] .= 0
∂ʳθ = @. ĉ * ∂ˣθ + ŝ * ∂ʸθ
uʳ = @. ĉ * uθ + ŝ * vθ



fig = Figure()
ax11 = Axis(fig[1, 1]; title="flux and gradient")
ax12 = Axis(fig[1, 2]; title="flux and gradient")
ax13 = Axis(fig[2, 1]; title="flux and gradient")
ax14 = Axis(fig[2, 2]; title="flux and gradient")
lines!(ax11, ∂ʳθ[:, 65], color = (:blue, 0.5) )
lines!(ax11, uʳ[:, 65], color = (:red, 0.5))
lines!(ax12, ∂ʳθ[65, :], color = (:blue, 0.5) )
lines!(ax12, uʳ[65, :], color  = (:red, 0.5))
scatter!(ax13, ∂ʳθ[:], uʳ[:])
r_flat_ind = sortperm(r[:])
r_flat = sort(r[:])
lines!(ax14, r_flat, ∂ʳθ[r_flat_ind], color = (:blue, 0.5))
lines!(ax14, r_flat, uʳ[r_flat_ind], color = (:red, 0.5))
display(fig)
