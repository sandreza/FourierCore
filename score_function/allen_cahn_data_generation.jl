using Random, HDF5, LinearAlgebra
include("allen_cahn_model.jl")
##
# Strong Nonlinearity
Random.seed!(1234)
parameters = (; N = 32, Ne = 2^7, ϵ² = 0.004, κ = 1e-3/4, U = 0.02, λ = 4e3)
pv, sf = allen_cahn(; parameters)
lr = linear_response_function(pv)
hsr = hack_score_response_function(pv, sf)
sr = score_response_function(pv, sf)
nr = numerical_response(; parameters, reset_scale = 100)
nonlinear_sr = copy(sr) # save for interactive
##
# Timestep is of size 1/N, thus we can scale by parameters.N to get one time unit
# divide by two to make it easier to train
scale_factor = 4
N, _, Ne, Nt = size(pv[:, :, :, 1:(parameters.N*10):end])
snapshots = copy(reshape(pv[:, :, :, 1:(parameters.N*10):end], (N, N, 1, Ne * Nt)))/scale_factor
N, _, Ne, Nt2 = size(pv[:, :, :, 1:parameters.N:end])
timeseries = copy(reshape(pv[:, :, :, 1:(parameters.N):end], (N, N, 1, Ne * Nt2)))/scale_factor
##
ilist = rand(1:Nt, 100)
jlist = rand(1:Nt, 100)
σ = maximum([norm(snapshots[:, :, 1, i][:] - snapshots[:, :, 1, j][:]) for i in ilist, j in jlist]) * 1.2
##
using HDF5
hfile = h5open("data_1.0_0.0_0.0_0.0.hdf5", "w")
hfile["snapshots"] = Float32.(snapshots)
hfile["timeseries"] = Float32.(timeseries)
hfile["snapshots shape"] = [N, N, 1, Ne, Nt]
hfile["timeseries shape"] = [N, N, 1, Ne, Nt2]
hfile["sigma"] = σ
hfile["scale factor"] = scale_factor
close(hfile)
##
a_linear_response = zeros(N, N, 100)
a_hack_score_response = zeros(N, N, 100)
a_numerical_response = zeros(N, N, 100)
for i in 1:100 
    a_linear_response[:, :, i] .= reshape(lr[i][:, 1], (N, N))
    a_hack_score_response[:, :, i] .= reshape(hsr[i][:, 1], (N, N))
    a_numerical_response[:, :, i] .= nr[:, :, parameters.N*(i-1)+1]
end
hfile = h5open("data_1.0_0.0_0.0_0.0_responses.hdf5", "w")
hfile["linear response"] = a_linear_response
hfile["exact score response"] = a_hack_score_response
hfile["perturbation response"] = a_numerical_response
close(hfile)
##
# No Nonlinearity
Random.seed!(1234)
parameters = (; N = 32, Ne = 2^7, ϵ² = 0.004, κ = 1e-3/4, U = 0.02, λ = 0* 4e3)
pv, sf = allen_cahn(; parameters)
lr = linear_response_function(pv)
hsr = hack_score_response_function(pv, sf)
sr = score_response_function(pv, sf)
nr = numerical_response(; parameters, reset_scale = 10)
##
scale_factor = 20
N, _, Ne, Nt = size(pv[:, :, :, 1:(N*10):end])
snapshots = copy(reshape(pv[:, :, :, 1:(N*10):end], (N, N, 1, Ne * Nt)))/scale_factor
N, _, Ne, Nt2 = size(pv[:, :, :, 1:N:end])
timeseries = copy(reshape(pv[:, :, :, 1:N:end], (N, N, 1, Ne * Nt2)))/scale_factor
##
ilist = rand(1:Nt, 100)
jlist = rand(1:Nt, 100)
σ = maximum([norm(snapshots[:, :, 1, i][:] - snapshots[:, :, 1, j][:]) for i in ilist, j in jlist]) * 1.2
##
using HDF5
hfile = h5open("data_0.0_0.0_0.0_0.0.hdf5", "w")
hfile["snapshots"] = Float32.(snapshots /2)
hfile["timeseries"] = Float32.(timeseries/2)
hfile["snapshots shape"] = [N, N, 1, Ne, Nt]
hfile["timeseries shape"] = [N, N, 1, Ne, Nt2]
hfile["sigma"] = σ
hfile["scale factor"] = scale_factor
close(hfile)
##
a_linear_response = zeros(N, N, 100)
a_hack_score_response = zeros(N, N, 100)
a_numerical_response = zeros(N, N, 100)
for i in 1:100 
    a_linear_response[:, :, i] .= reshape(lr[i][:, 1], (N, N))
    a_hack_score_response[:, :, i] .= reshape(hsr[i][:, 1], (N, N))
    a_numerical_response[:, :, i] .= nr[:, :, N*(i-1)+1]
end
hfile = h5open("data_0.0_0.0_0.0_0.0_responses.hdf5", "w")
hfile["linear response"] = a_linear_response
hfile["exact score response"] = a_hack_score_response
hfile["perturbation response"] = a_numerical_response
close(hfile)