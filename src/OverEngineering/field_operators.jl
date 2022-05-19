import Base: *, +

using FFTW

abstract type AbstractOperator end

struct PointWiseSpectralOperator{O} <: AbstractOperator
    operation::O
end

*(a::AbstractField, b::AbstractField) = a.data .* b.data
+(a::AbstractField, b::AbstractField) = a.data .+ b.data

function (operator::PointWiseSpectralOperator)(a::ScalarField)
    â = fft(a.data) 
    data = operator.operation .* â
    return ifft(data)
end

function (operator::PointWiseSpectralOperator)(a::AbstractArray)
    â = fft(a)
    data = operator.operation .* â
    return ifft(data)
end