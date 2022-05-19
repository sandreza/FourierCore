abstract type AbstractField end

struct ScalarField{D,G,U} <: AbstractField
    data::D
    grid::G
    utils::U
end

function ScalarField(data, grid)
    ℱ = plan_fft!(data)
    ℱ⁻¹ = plan_ifft!(data)
    utils = (; forward=ℱ, backward=ℱ⁻¹)
    return ScalarField(data, grid, utils)
end

struct VectorField{D, G, U} <: AbstractField
    data::D
    grid::G
    utils::U
end

struct StateField{D, G, U} <: AbstractField
    data::D
    grid::G
    utils::U
end

