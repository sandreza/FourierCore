function linear_response_function(pixel_value; skip = 32, dt = 1/32)
    N = size(pixel_value)[1]
    Ne = size(pixel_value)[3]
    pv = pixel_value .- mean(pixel_value);
    endindex = size(pv[:, :, :, 1:skip:end])[end]÷2
    ts = skip * dt .* collect(0:endindex-1)
    spatio_temporal_autocov = zeros(N, N, endindex);
    @info "Computing Space-Time Autocorrelation"
    for i in ProgressBar(1:Ne)
        spatio_temporal_autocov .+= real.(ifft(fft(pv[:, :, i, 1:skip:end]) .* ifft(pv[:, :, i, 1:skip:end]))[:, :, 1:endindex]/Ne)
    end
    @info "Computing Covariance Matrix"
    covmat = zeros(N^2, N^2, endindex);
    for k in ProgressBar(1:endindex)
        for i in 1:N^2
            ii = (i-1) % N + 1
            jj = (i-1) ÷ N + 1
            @inbounds covmat[:, i, k] .= circshift(spatio_temporal_autocov[:, :, k], (ii-1, jj-1))[:]
        end
    end
    @info "Inverting Covariance Matrix"
    C⁻¹ = inv(covmat[:, :, 1])
    ##
    @info "Computing Response Function"
    return response_function = [covmat[:, :, i] * C⁻¹ for i in ProgressBar(1:endindex)]
end