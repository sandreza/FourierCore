include("initialize_fields.jl")

f_amp = 300

forcing_amplitude = f_amp * (N / 2^7)^2 # due to FFT nonsense
constants = (; forcing_amplitude=forcing_amplitude, ϵ=ϵ)
parameters = (; auxiliary, operators, constants)
include("lagrangian_eulerian_ensemble.jl")

# lagrangian further ensemble averaging
skip = round(Int,decorrelation_index / mod_index)
# start_index
si = Int(maximum([argmax(lagrangian_list) % skip, skip]))
# end index
ei = floor(Int, (length(lagrangian_list) - si + 1) / skip) 
formatted_lagrangian_list = [lagrangian_list[si+(i-1)*skip:si+i*skip-1] for i in 1:ei]

# eulerian further ensemble averaging 
skip = round(Int,decorrelation_index2 / mod_index)
# start_index
si = Int(maximum([argmax(eulerian_list) % skip, skip]))
# end index
ei = floor(Int, (length(eulerian_list) - si + 1) / skip) 
formatted_eulerians_list = [eulerian_list[si+(i-1)*skip:si+i*skip-1] for i in 1:ei]
scatter(mean(formatted_lagrangian_list))
scatter!(mean(formatted_eulerian_list))

#=
# = "/storage5/NonlocalPassiveTracers/Current/" * "prototype.hdf5"
directory = "/storage5/NonlocalPassiveTracers/Current/"
fid = h5open(directory * "prototype2.hdf5", "w")
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
fid["stream function"] = real.(Array(ifft(ψ)))
fid["lagrangian decorrelation"] = mean(formatted_lagrangian_list)
fid["lagrangian decorrelation unprocessesed"] = lagrangian_list 
fid["eulerian decorrelation unprocessesed"] = eulerian_list
fid["eulerian decorrelation"] = mean(formatted_eulerian_list)
fid["kinetic energy evolution"] = ke_list
fid["times output"] = tlist
fid["domain size x"] = Ω[1].b - Ω[1].a
fid["domain size y"] = Ω[2].b - Ω[2].a
close(fid)
=#
