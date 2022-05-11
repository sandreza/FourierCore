using NCDatasets, DataStructures
ds = Dataset("diffusivity_2_random_phase.nc", "c")
initial_data = θ_A
final_data = ld
ensemble_data = θ_F

defDim(ds, "x", size(initial_data)[1])
defDim(ds, "y", size(initial_data)[2])

v = defVar(ds, "t0", Float32, ("x", "y"), attrib=OrderedDict(
    "units" => " "))
v.attrib["comments"] = "The initial tracer concentration"
v[:, :] = initial_data

w = defVar(ds, "kappa_2e-13", Float32, ("x", "y"), attrib=OrderedDict(
    "units" => " "))
w.attrib["comments"] = "Tracer concentration at t=20"
w[:, :] = final_data

we = defVar(ds, "kappa_2e-8", Float32, ("x", "y"), attrib=OrderedDict(
    "units" => " "))
we.attrib["comments"] = "Tracer concentration at t=20"
we[:, :] = ensemble_data

close(ds)
