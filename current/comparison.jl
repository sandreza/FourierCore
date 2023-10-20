using GLMakie, HDF5

# Grab propagator version
directory = "/storage5/NonlocalPassiveTracers/Current/"
base_name = "ens_full_propagator_reguralized_" # "full_propagator_"
ii = 3 # forcing
kk = 1 # hypo
jj = 1 # hyper
filename = base_name * string(ii) * "_" * string(jj) * "_" * string(kk)
fid = h5open(directory * filename * ".hdf5", "r")
𝒦 = read(fid["regularized space time kernel"])# read(fid["space time kernel"])
scale = abs.(fft(𝒦[:, 1]) ./ fft(𝒦[:, 1])[1])
ts = read(fid["space time kernel timelist"])
close(fid)

# Grab indirect way
base_name = "even_higher_frequency_general_case_"
ii = 3 # forcing
kk = 1 # hypo
jj = 1 # hyper
filename = base_name * string(ii) * "_" * string(jj) * "_" * string(kk)
fid = h5open(directory * filename * ".hdf5", "r")
𝒦st = read(fid["space time kernel"])
ωs = reverse(read(fid["time dependent frequences"]))
Tfundamental = 2π / ωs[2]
Δtst = Tfundamental / (size(𝒦st)[2]-1) 
close(fid)


# Grab production version
directory = "/storage5/NonlocalPassiveTracers/Current/"
base_name = "production_"
ii = 4 # forcing, changed the ordering ... 
kk = 1 # hypo
jj = 1 # hyper
filename = base_name * string(ii) * "_" * string(jj) * "_" * string(kk)

hfile = h5open(directory * filename * ".hdf5", "r")
k0 = read(hfile["large scale effective diffusivity"])
ks = read(hfile["diffusivity kernel fourier"])
𝒦2 = vcat(k0, ks)
close(hfile)

##
labelsize = 40
ms = 20
options = (; titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)
wavenumbers = collect(0:0.5:64)
new𝒦 = real.(fft(circshift(sum(𝒦, dims = 2)[:], -64) * ts[2])) # ./ scale

indirect𝒦 = real.(fft(circshift(sum(𝒦st, dims = 2)[:], -41)))

fig = Figure() 
ax11 = Axis(fig[1,1]; xlabel = "wavenumber", ylabel = "eigenvalue", options...)
ax12 = Axis(fig[1,2]; xlabel = "wavenumber", ylabel = "eigenvalue", options...)
scatter!(ax11, wavenumbers[1:65], new𝒦[1:65], label = "propagator time average", color = (:red, 0.5), markersize = ms)
scatter!(ax11, wavenumbers[1:40], 𝒦2[1:40], label = "indirect", color = (:blue, 0.5), markersize = ms)
axislegend(ax11, position=:rt, framecolor=(:grey, 0.5), patchsize=(30, 30), markersize=50, labelsize=20)

scatter!(ax12, wavenumbers[1:40], 𝒦2[1:40], label = "indirect", color = (:blue, 0.5), markersize = ms)
scatter!(ax12, wavenumbers[1:65], indirect𝒦[1:65], label = "indirect time averaged", color = (:orange, 0.5), markersize = ms)
axislegend(ax12, position=:rt, framecolor=(:grey, 0.5), patchsize=(30, 30), markersize=50, labelsize=20)
display(fig)