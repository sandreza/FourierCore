using FFTW
using Statistics

function dealias(u)
    N = length(u)
    Nd = round(Int, N/3)
    
    v = u
    v[Nd+2:N-Nd] .= 0

    return v
end

function domain(L, N)
    x = L*(0:N-1)'/N .- L/2
    k = (2*pi/L)*[0:N/2; -N/2+1:-1]'

    return x,k
end

function field2vector(f,N,s)
    Nd = round(Int, N/3)
    v = imag(f[2:Nd+1])

    if !s
        v = [v;real(f[2:Nd+1]);real(f[1])]
    end

    return v
end

function vector2field(v,N,s)
    Nd = round(Int, N/3)

    f = zeros(N,1)*im
    f[2:Nd+1] = v[1:Nd]*im
    if !s
        f[2:Nd+1] = f[2:Nd+1] + v[Nd+1:2*Nd]
        f[1] = v[2*Nd+1]
    end

    f[N÷2+2:end] = conj(f[N÷2:-1:2])

    return f
end

# Based on Kassam & Trefethen (2005)
function KSE_integrate(L, dt_ref, t_max, t_store, u0, s, zero_mean=true)
    ## adjust time step size
    N_steps = floor(Int, t_max/dt_ref) + 1
    dt = t_max / N_steps

    ## grid and initial condition
    N = length(u0)
    _,k = domain(L, N)

    v = dealias(u0)

    if s
        v = im*imag(v)
    end

    ## precompute ETDRK4 scalars
    Linear = k.^2 - k.^4
    
    E = exp.(dt*Linear)
    E2 = exp.(dt*Linear/2)
    
    M = 32
    r = exp.(im*pi*((1:M)' .- 0.5)/M)
    LR = dt*repeat(transpose(Linear),1,M) + repeat(r,N,1)
    Q = dt*real(mean((exp.(LR/2) .- 1)./LR, dims = 2))'
    f1 = dt*real(mean((-4 .- LR + exp.(LR).*(4 .- 3*LR+LR.^2))./LR.^3, dims = 2))'
    f2 = dt*real(mean((2 .+ LR + exp.(LR).*(-2 .+ LR))./LR.^3, dims = 2))'
    f3 = dt*real(mean((-4 .- 3*LR - LR.^2 + exp.(LR).*(4 .- LR))./LR.^3, dims = 2))'
    
    ## time-stepping loop:
    g = -0.5im*k

    if t_store == 0
        n = 1
    else
        n = floor(Int, t_store/dt)

        snapshot = field2vector(v,N,s)

        snapshots = zeros(length(snapshot), N_steps÷n)
        snapshots[:,1] = snapshot

        t_grid = zeros(N_steps÷n)
    end

    
    IF = plan_ifft(v, flags=FFTW.MEASURE)

    F! = plan_fft!(v, flags=FFTW.MEASURE)
    IF! = plan_ifft!(v, flags=FFTW.MEASURE)

    t = 0;
    tic = time()
    for q = 1:N_steps
        t = t+dt
        
        Nv = g.*dealias(F!*((IF*v).^2))

        a = E2.*v + Q.*Nv
        Na = g.*dealias(F!*((IF*a).^2))
        
        b = E2.*v + Q.*Na
        Nb = g.*dealias(F!*((IF!*b).^2))
        
        c = E2.*a + Q.*(2*Nb-Nv)
        Nc = g.*dealias(F!*((IF!*c).^2))
        
        v = E.*v + Nv.*f1 + 2*(Na+Nb).*f2 + Nc.*f3

        # maintain conjugate symmetry
        v[N÷2+2:end] = conj(v[N÷2:-1:2])

        if s
            v = im*imag(v)
        end

        if zero_mean
            v[1] = 0
        else
            v[1] = real(v[1])
        end
        
        if rem(q,n) == 0 && t_store != 0
            snapshots[:,q÷n] = field2vector(v,N,s)
            t_grid[q÷n] = t
        end
    end
    println("Time elapsed: ", time()-tic, " seconds")

    if t_store == 0
        snapshots = field2vector(v,N,s)
        t_grid = t
    end

    return snapshots,t_grid
end