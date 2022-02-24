using KernelAbstractions
# random_phase_kernel!(field, @Const(A), @Const(𝓀ˣ), @Const(𝓀ʸ), @Const(x), @Const(y), @Const(φ), Nx, Ny)

# only expect performance on the gpu
@kernel function random_phase_kernel!(field, @Const(A), @Const(𝓀ˣ), @Const(𝓀ʸ), @Const(x), @Const(y), @Const(φ), Nx, Ny)
    i, j = @index(Global, NTuple)

    tmp_sum = zero(eltype(field)) # create temporary array
    for ii in 1:Nx, jj in 1:Ny
        tmp_sum += A[ii, jj] * cos(𝓀ˣ[ii] * x[i] + 𝓀ʸ[jj] * y[j] + φ[ii, jj])
    end

    field[i, j] = tmp_sum
end

function random_phase_check!(field, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
    kernel! = random_phase_kernel!(CPU(), 1)
    Nx = length(𝓀ˣ)
    Ny = length(𝓀ʸ)
    kernel!(field, A, 𝓀ˣ, 𝓀ʸ, x, y, φ, Nx, Ny, ndrange = size(field))
end

event = random_phase_check!(field, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
wait(event)
##
tic = Base.time()
for i in 1:100
    event = random_phase_check!(field, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
    wait(event)
end
toc = Base.time()
println("took ", toc - tic, " seconds ")

tic = Base.time()
for i in 1:100
    random_phase(field, A, 𝓀ˣ, 𝓀ʸ, x, y, φ)
end
toc = Base.time()
println("took ", toc - tic, " seconds ")
