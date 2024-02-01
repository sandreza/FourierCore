using Random, HDF5, LinearAlgebra
include("allen_cahn_model.jl")
##
# Strong Nonlinearity
Random.seed!(1234)
parameters = (; N = 32, Ne = 2^7, ϵ² = 0.004, κ = 1e-3/4, U = 0.02, λ = 4e3)
sf = allen_cahn_score(; parameters)

hfile = h5open("data_1.0_0.0_0.0_0.0.hdf5", "r")
tshap = read(hfile["timeseries shape"])
ppv = read(hfile["timeseries"])
close(hfile)
pv = reshape(ppv, (tshap[1], tshap[2], tshap[4], tshap[5])) * 4
sr = score_function(pv, sf; skip = 1)

hfile = h5open("data_1.0_0.0_0.0_0.0_no_hack_response.hdf5", "w")
hfile["score response"] = sr
close(hfile)

##
parameters = (; N = 32, Ne = 2^5, ϵ² = 0.004, κ = 1e-3/4, U = 0.02, λ = 0* 4e3)
sf = allen_cahn_score(; parameters)

hfile = h5open("data_0.0_0.0_0.0_0.0.hdf5", "r")
tshap = read(hfile["timeseries shape"])
scale_factor = read(hfile["scale factor"])
ppv = read(hfile["timeseries"])
close(hfile)
pv = reshape(ppv, (tshap[1], tshap[2], tshap[4], tshap[5])) * scale_factor * 2
sr = score_function(pv, sf; skip = 1)

hfile = h5open("data_0.0_0.0_0.0_0.0_no_hack_response.hdf5", "w")
hfile["score response"] = sr
close(hfile)