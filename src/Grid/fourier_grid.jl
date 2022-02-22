import Base: getindex
import FourierCore.Domain: Torus, Circle
import Base: size

export FourierGrid
export create_grid, getindex, size

struct FourierGrid{𝒢,𝒲,𝒟}
    nodes::𝒢
    wavenumbers::𝒲
    domain::𝒟
end


getindex(f::FourierGrid, i) = f.nodes[i]

function FourierGrid(grid_points, Ω::Circle; arraytype = Array) # perhaps change to match the product domain
    @assert length(grid_points) == 1
    grid = arraytype(nodes(grid_points, a = Ω.a, b = Ω.b))
    k⃗ = arraytype(wavenumbers(grid_points, L = Ω.b - Ω.a))
    return FourierGrid([grid], [k⃗], Ω)
end

"""
FourierGrid(grid_points, Ω::ProductDomain, arraytype=Array)
# Description
Create a numerical grid with grid_points resolution in the domain Ω \n 
Only works for fully periodic grids at the moment
# Arguments
- `grid_points`: tuple | a tuple of ints in each direction for product domain
- `Ω`: ProductDomain   | a product domain object
# Keyword Arguments
- arraytype = Array
# Return
A Fourier Grid object
"""
function FourierGrid(grid_points::Tuple, Ω::Torus; arraytype = Array) # change to be fully general later
    @assert length(grid_points) == length(Ω.domains)
    grid = []
    k⃗ = []
    for (i, domain) in enumerate(Ω.domains)
        L = domain.b - domain.a
        reshape_dims = appropriate_dims(length(Ω.domains), i, grid_points[i])
        push!(grid, arraytype(reshape(nodes(grid_points[i], a = domain.a, b = domain.b), reshape_dims)))
        push!(k⃗, arraytype(reshape(wavenumbers(grid_points[i], L = L), reshape_dims)))
    end
    return FourierGrid(Tuple(grid), Tuple(k⃗), Ω)
end

FourierGrid(grid_points::Int, Ω::Torus; arraytype = Array) = FourierGrid(Tuple(fill(grid_points, ndims(Ω))), Ω, arraytype = arraytype)
FourierGrid(grid_points::Array, Ω::Torus; arraytype = Array) = FourierGrid(Tuple(grid_points), Ω, arraytype = arraytype)

function Base.show(io::IO, F::FourierGrid)
    println("domain:", F.domain)
    print("gridpoints:")
    if typeof(F.nodes[1]) <: AbstractArray
        for (i, grid) in enumerate(F.nodes)
            print(length(grid))
            if i != length(F.nodes)
                printstyled(io, "×")
            end
        end
    else
        print(length(F.nodes))
    end
end

size(f::FourierGrid) = length.(f.nodes)
