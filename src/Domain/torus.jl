import LinearAlgebra: ×
import Base: ^, getindex, ndims

export Circle, S¹, Torus
export ×, ndims, info


struct Circle{ℱ}
    a::ℱ
    b::ℱ
end

function Circle(b::FT) where {FT}
    return Circle(FT(0), b)
end

function Circle()
    return Circle(2π)
end

S¹ = Circle

function Base.show(io::IO, Ω::Circle)
    printstyled(io, "[", color = 226)
    a = @sprintf("%.3f", Ω.a)
    b = @sprintf("%.3f", Ω.b)
    printstyled("$a, $b", color = 7)
    printstyled(io, ")", color = 226)
end

struct Torus{DT}
    domains::DT
end

function Base.show(io::IO, Ω::Torus)
    for (i, domain) in enumerate(Ω.domains)
        print(domain)
        if i != length(Ω.domains)
            printstyled(io, "×", color = 118)
        end
    end
end

function ndims(Ω::Circle)
    return 1
end

function ndims(Ω::Torus)
    return length(Ω.domains)
end

getindex(t::Torus, i) = t.domains[i]

# Algebra
×(arg1::Circle, arg2::Circle) = Torus((arg1, arg2))
×(args::Torus, arg2::Circle) = Torus((args.domains..., arg2))
×(arg1::Circle, args::Torus) = Torus((arg1, args.domains...))
×(arg1::Torus, args::Torus) = Torus((arg1.domains..., args.domains...))
×(args::Torus) = Torus(args...)
# Exponentiation
^(circle::Circle, n::Int) = ^(circle, Val(n))
^(circle::Circle, ::Val{1}) = circle
^(circle::Circle, ::Val{2}) = circle × circle
^(circle::Circle, ::Val{n}) where {n} = circle × ^(circle, Val(n - 1))


function info(Ω::Torus)
    println("This is a ", ndims(Ω), "-dimensional Torus.")
    print("The domain is ")
    println(Ω, ".")
    for (i, domain) in enumerate(Ω.domains)
        domain_string = "periodic"
        length = @sprintf("%.2f ", domain.b - domain.a)
        println("The dimension $i domain is ", domain_string, " with length ≈ ", length)

    end
    return nothing
end
