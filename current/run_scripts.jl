tic = Base.time()
# These first three scripts define the fields, constants, and operators
include("initialize_fields.jl")
include("initialize_constants.jl")
include("initialize_operators.jl")
# the parameters named tuple is defined by this point

include("initialize_ensembles.jl")
include("diffusivity_kernel.jl")
include("large_scale.jl")
include("large_scale_time_dependent.jl")
toc = Base.time()

println("Running the scripts takes ", (toc - tic ) /60, " minutes to run for $N_ens ensemble members")