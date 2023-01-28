tic = Base.time()
include("initialize_fields.jl")
include("initialize_ensembles.jl")
include("diffusivity_kernel.jl")
include("large_scale.jl")
incldue("large_scale_time_dependent.jl")
toc = Base.time()

println("Running the scripts takes ", (toc - tic ) /60, " minutes to run for $N_nes ensemble members")