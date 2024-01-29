using GLMakie


include("KSE_integrate.jl")


N = 64
L = 34 # 40.95
symm = false
dt = 0.5
T = 100000
tstore = 2.

x,_ = domain(L, N)
u = sin.(2*pi*x/L) + 0.1*cos.(10*pi*x/L)

u = dealias(fft(u))
u[1] = 0 # Ensure zero mean

tic = time()
u,t = KSE_integrate(L, dt, T, tstore, u, symm, true)
toc = time()
println("time for sim ", toc - tic, " seconds")
uu = zeros(length(t),N)
E  = zeros(length(t))

for i=1:length(t)
    u_spectral = vector2field(u[:,i],N,symm)
    u_physical = real(ifft(u_spectral))
    uu[i,:] = u_physical
    E[i] = sqrt(sum(u_physical.^2))
end

lines(t,E)