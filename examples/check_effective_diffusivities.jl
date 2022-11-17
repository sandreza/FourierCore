using HDF5

kefflist = []
λlist = [0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
for λ ∈ λlist
    file = "effective_diffusivities_amp1_λ_" * string(λ) * ".hdf5"
    fid = h5open(file, "r")

    keff = read(fid["effective_diffusivities"]["1"])
    push!(kefflist,keff)
    close(fid)
end
