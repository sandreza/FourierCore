using DifferentialEquations, BenchmarkTools, LinearAlgebra

function lorenz(du, u, p, t)
    du[1] = 10.0(u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end

function σ_lorenz(du, u, p, t)
    du[1] = 3.0
    du[2] = 3.0
    du[3] = 3.0
end

prob_sde_lorenz = SDEProblem(lorenz, σ_lorenz, [4.445219220981931, 6.258499227223414, 14.862969602338891], (0.0, 10.0))
sol = solve(prob_sde_lorenz, SOSRI(), save_everystep = false, save_end = true, dt = 0.15, adaptive = false)

solver_methods = [SOSRI(), EM(), EulerHeun(), RKMil(), WangLi3SMil_A(), SRI(), SRIW2(), SRA(), SOSRA()]
stable_solver_methods = []
for solver_method in solver_methods
    prob_sde_lorenz = SDEProblem(lorenz, σ_lorenz, [4.445219220981931, 6.258499227223414, 14.862969602338891], (0.0, 10.0))
    sol = solve(prob_sde_lorenz, solver_method, save_everystep = true, save_end = true, dt = 0.001, adaptive = true, abstol = 1e1, reltol = 1e1)

    println("------------")
    println("For solver ", solver_method, " we compute ")
    println("solution at final timestep ", sol.u[end])
    println("The final time is ", sol.t[end])
    println("largest timestep size ", maximum(sol.t[2:end] - sol.t[1:end-1]))
    println("smallest timestep size ", minimum(sol.t[2:end] - sol.t[1:end-1]))
    println("-----------")
    if sol.t[end] ≈ 10.0
        push!(stable_solver_methods, solver_method)
    end
end

benchmarks = []
benchy = @benchmark solve(prob_sde_lorenz, SOSRI(), save_everystep = false, save_end = false, dt = 0.08, adaptive = false)
benchy2 = @benchmark solve(prob_sde_lorenz, SOSRA(), save_everystep = false, save_end = false, dt = 0.08, adaptive = false)
