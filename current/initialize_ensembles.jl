@show "initializing ensembles"
include("lagrangian_eulerian_ensemble.jl")

# lagrangian further ensemble averaging
skipL = round(Int,decorrelation_index / mod_index)
# start_index
si = Int(maximum([argmax(lagrangian_list) % skipL, skipL]))
# end index
ei = floor(Int, (length(lagrangian_list) - si + 1) / skipL) 
formatted_lagrangian_list = [lagrangian_list[si+(i-1)*skipL:si+i*skipL-1] for i in 1:ei]

# eulerian further ensemble averaging 
skip = round(Int,decorrelation_index2 / mod_index)
# start_index
si = Int(maximum([argmax(eulerian_list) % skip, skip]))
# end index
ei = floor(Int, (length(eulerian_list) - si + 1) / skip) 
formatted_eulerian_list = [eulerian_list[si+(i-1)*skip:si+i*skip-1] for i in 1:ei]

#=
fig = Figure()
ax = Axis(fig[1, 1]; xlabel="time", ylabel="Decorrelation")
scatter!(ax, tlist[1:skipL] .- tlist[1], mean(formatted_lagrangian_list); label = "lagrangian")
scatter!(ax, tlist[1:skip] .- tlist[1], mean(formatted_eulerian_list); label = "eulerian")
axislegend(ax; position = :rt)
display(fig)
=#

# = "/storage5/NonlocalPassiveTracers/Current/" * "proto_default_case.hdf5"
directory = "/storage5/NonlocalPassiveTracers/Current/"
fid = h5open(directory * filename * ".hdf5", "w")
fid["forcing amplitude"] = f_amp
fid["Nx"] = N
fid["Ny"] = N
fid["Nensemble"] = N_ens
fid["nu"] = ν^dissipation_power
fid["nu harmonic power"] = dissipation_power 
fid["kappa"] = κ
fid["hypoviscocity"] = ν_h^hypoviscocity_power
fid["hypoviscosity power"] = hypoviscocity_power
fid["vorticity"] = real.(Array(ζ))
P⁻¹ * ψ
fid["stream function"] = real.(Array(ψ))
fid["lagrangian decorrelation"] = mean(formatted_lagrangian_list)
fid["lagrangian decorrelation unprocessesed"] = lagrangian_list 
fid["eulerian decorrelation unprocessesed"] = eulerian_list
fid["eulerian decorrelation"] = mean(formatted_eulerian_list)
fid["kinetic energy evolution"] = ke_list
fid["times output decorrelation case"] = tlist
fid["domain size x"] = Ω[1].b - Ω[1].a
fid["domain size y"] = Ω[2].b - Ω[2].a
fid["dt"] = Δt
close(fid)

@show "done initializing ensembles"
