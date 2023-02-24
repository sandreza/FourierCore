f_amps = [50, 150, 300, 450, 750]
νs = [sqrt(1e-5 / 2)]
ν_hs = [sqrt(1e-3), sqrt(1e-4)]
tic = Base.time()

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

ii = 4
jj = 1
kk = 2
f_amp = f_amps[ii]
ν = νs[jj]
ν_h = ν_hs[kk]
include("compute_cases.jl")
toc = Base.time()
println("Elapsed time: ", (toc - tic) / (60 * 60), " hours")