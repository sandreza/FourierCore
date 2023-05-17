f_amps = [1, 10, 50, 300, 750] # [750, 300, 50, 10, 1]
νs = [sqrt(1e-5 / 2)]
ν_hs = [sqrt(1e-3), sqrt(1e-4), sqrt(1e-2)]
tic = Base.time()

N = 2^7
N_ens = 2^7 # 2^7
Ns = (N, N, N_ens)
base_name = "production_"


ii = 1
jj = 1
kk = 1
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
include("compute_cases.jl")
toc = Base.time()
println("Elapsed time: ", (toc - tic) / (60 * 60), " hours")

ii = 4
jj = 1
kk = 1
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
include("compute_cases.jl")
toc = Base.time()
println("Elapsed time: ", (toc - tic) / (60 * 60), " hours")


ii = 3
jj = 1
kk = 1
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
include("compute_cases.jl")
toc = Base.time()
println("Elapsed time: ", (toc - tic) / (60 * 60), " hours")


ii = 4
jj = 1
kk = 1
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
include("compute_cases.jl")
toc = Base.time()
println("Elapsed time: ", (toc - tic) / (60 * 60), " hours")

ii = 5
jj = 1
kk = 1
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
include("compute_cases.jl")
toc = Base.time()
println("Elapsed time: ", (toc - tic) / (60 * 60), " hours")
