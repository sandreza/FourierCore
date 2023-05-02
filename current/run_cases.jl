f_amps = [50, 150, 300, 450, 750, 0.1, 1, 10, 0.01]
νs = [sqrt(1e-5 / 2)]
ν_hs = [sqrt(1e-3), sqrt(1e-4), sqrt(1e-2)]
tic = Base.time()

base_name = "case_"
N = 2^7
N_ens = 2^5 # 2^7
Ns = (N, N, N_ens)

jj = 1
ii = 8
kk = 3
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
include("compute_cases.jl")
toc = Base.time()
println("Elapsed time: ", (toc - tic) / (60 * 60), " hours")
#=
ii = 7
kk = 1
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
include("compute_cases.jl")
toc = Base.time()
println("Elapsed time: ", (toc - tic) / (60 * 60), " hours")

ii = 8
kk = 1
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
include("compute_cases.jl")
toc = Base.time()
println("Elapsed time: ", (toc - tic) / (60 * 60), " hours")
=#
#=
ii = 9
kk = 1
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
include("compute_cases.jl")
toc = Base.time()
println("Elapsed time: ", (toc - tic) / (60 * 60), " hours")
=#
#=
ii = 8
kk = 2
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
include("compute_cases.jl")
toc = Base.time()
println("Elapsed time: ", (toc - tic) / (60 * 60), " hours")

ii = 9
kk = 2
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
include("compute_cases.jl")
toc = Base.time()
println("Elapsed time: ", (toc - tic) / (60 * 60), " hours")

##
ii = 6
kk = 1
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
include("compute_cases.jl")
toc = Base.time()
println("Elapsed time: ", (toc - tic) / (60 * 60), " hours")

ii = 7
kk = 1
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
include("compute_cases.jl")
toc = Base.time()
println("Elapsed time: ", (toc - tic) / (60 * 60), " hours")

ii = 8
kk = 1
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
include("compute_cases.jl")
toc = Base.time()
println("Elapsed time: ", (toc - tic) / (60 * 60), " hours")

ii = 9
kk = 1
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
include("compute_cases.jl")
toc = Base.time()
println("Elapsed time: ", (toc - tic) / (60 * 60), " hours")
=#
#=
ii = 1
jj = 1
kk = 1
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
include("compute_cases.jl")

ii = 2
jj = 1
kk = 1
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
include("compute_cases.jl")


ii = 3
jj = 1
kk = 1
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
include("compute_cases.jl")

ii = 4
jj = 1
kk = 1
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
include("compute_cases.jl")


ii = 1
jj = 1
kk = 2
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
include("compute_cases.jl")

ii = 2
jj = 1
kk = 2
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
include("compute_cases.jl")

ii = 3
jj = 1
kk = 2
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
include("compute_cases.jl")

ii = 4
jj = 1
kk = 2
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
include("compute_cases.jl")

toc = Base.time()
println("Elapsed time: ", (toc - tic) / (60 * 60), " hours")
=#

#=
tic = Base.time()
ii = 5
jj = 1
kk = 1
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
include("compute_cases.jl")
=#

#=
ii = 4
jj = 1
kk = 2
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
include("compute_cases.jl")
toc = Base.time()
println("Elapsed time: ", (toc - tic) / (60 * 60), " hours")
=#