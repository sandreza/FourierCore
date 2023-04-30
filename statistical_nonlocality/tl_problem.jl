using LinearAlgebra, Random, ProgressBars
# GLMakie, Statistics, Random, Distributions

k = 1.0 # wavenumber
κ = 0.01 # diffusivity
λ = 0.0 # relaxation rate 
γ = 1.0 # ou relaxation: default = 1.0
ϵ = sqrt(2) # noise strength: default = √2

N = 5 # number of markov states - 1, numerically unstable for large N

# Discrete OU process ALA Charlie Doering
function ou_transition_matrix(n)
    Mⱼₖ = zeros(n + 1, n + 1)
    δ(j, k) = (j == k) ? 1 : 0

    for j in 0:n, k in 0:n
        jj = j + 1
        kk = k + 1
        Mⱼₖ[jj, kk] =
            (-n * δ(j, k) + k * δ(j + 1, k) + (n - k) * δ(j - 1, k)) / 2
    end
    return Mⱼₖ
end

# construct markov approximation 
Δx = 2 / √N
uₘ = 1 / sqrt(γ * 2/ϵ^2) * [Δx * (i - N / 2) for i in 0:N]
Q = ou_transition_matrix(N) .* γ
Λ, V = eigen(Q)
V⁻¹ = inv(V)

# define the effective diffusivity as the appropriate schur-complement
U = V * Diagonal(uₘ) * V⁻¹
vtop = U[end, 1:end-1]
vbottom = U[1:end-1, end]
##
keff = Float64[]
for k in 1:7
    vbot = im * k * U[1:end-1, 1:end-1] + Diagonal(Λ[1:end-1] .- λ .- κ * k^2)
    𝒦ₘ = -real(vtop' * (vbot \ vbottom))
    push!(keff, 𝒦ₘ)
end
##
# Not sure if this is totally sensible
dl = reverse([1.0 * n for n in 1:N])
du = [1.0 for n in 1:N]
d = zeros(N + 1)
Uₕ = 1 / sqrt(γ * 2/ϵ^2) .* Tridiagonal(dl, d, du) # Hermite Polynomial U, position operator in spectral space

ll, vv = eigen(Array(Uₕ)) # nodal positions
QH = vv * Diagonal(Λ) * inv(vv) # nodal space matrix
vtop = Uₕ[end, 1:end-1]
vbottom = Uₕ[1:end-1, end]
vbot = im * k * Uₕ[1:end-1, 1:end-1] + Diagonal(Λ[1:end-1] .- λ .- κ * k^2)
𝒦ₕ = -real(vtop' * (vbot \ vbottom))