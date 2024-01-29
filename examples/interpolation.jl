quick = 1
# assume 2D 
function quick_interpolation(field)
    afield = Array(field)
    Nx, Ny = size(field)
    @assert Nx == Ny
    N = Nx
    interpolated_field = zeros(ComplexF64, 512, 512)
    fouriedfield = fft(afield)
    if N > 512
        indices1 = 1:div(512, 2)+1
        indices2 = N-div(512, 2)+2:N
        indices = [indices1..., indices2...]
        vfield = view(fouriedfield, indices, indices)
        interpolated_field .= vfield
        return real.(ifft(interpolated_field))
    elseif N < 512
        indices1 = 1:div(N, 2)+1
        indices2 = 512-div(N, 2)+2:512
        indices = [indices1..., indices2...]
        vfield = view(interpolated_field, indices, indices)
        vfield .= fouriedfield
        return real.(ifft(interpolated_field))
    else
        return real.(afield)
    end

end

# assume 1d
function quick_interpolation_1d(field; M = 256)
    afield = Array(field)
    Nx = size(field)[1]
    N = Nx
    interpolated_field = zeros(ComplexF64, M)
    fouriedfield = fft(afield)
    if N > M
        indices1 = 1:div(M, 2)+1
        indices2 = N-div(M, 2)+2:N
        indices = [indices1..., indices2...]
        vfield = view(fouriedfield, indices)
        interpolated_field .= vfield
        return real.(ifft(interpolated_field))
    elseif N < M
        indices1 = 1:div(N, 2)+1
        indices2 = M-div(N, 2)+2:M
        indices = [indices1..., indices2...]
        vfield = view(interpolated_field, indices)
        vfield .= fouriedfield
        return real.(ifft(interpolated_field))
    else
        return real.(afield)
    end

end