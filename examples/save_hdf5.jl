using HDF5

fid = h5open("random_phase.hdf5", "w")
fid["t0"] = θ_A
fid["t5"] = θ_F
fid["ensemble"] = θ̅_F
close(fid)

#=
newfid = h5open("random_phase.hdf5", "r")
array1 = read(newfid["t0"])
=#