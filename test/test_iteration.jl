# CODE STATUS: TESTS NOT DOCUMENTED
using SparseArrays
import MultiScaleOT as MOT 
import MultiScaleOT: AbstractMeasure, GridMeasure, npoints, normalize!
import DomDecOT as DD
import DomDecOT: DomDecPlan
import Random
import StatsBase: mean

function random_setup_domdec(N, cellsize)
    x1 = collect(1:N)
    X = MOT.flat_grid(x1)
    Y = MOT.flat_grid(x1)
    μ = rand(N) .+ 1e-1; normalize!(μ)
    ν = rand(N) .+ 1e-1; normalize!(ν)

    mu = MOT.GridMeasure(X, μ, (N,))
    nu = MOT.GridMeasure(Y, ν, (N,))

    gamma0 = ν .* μ'
    basic_cells, _ = DD.get_cells((N,), cellsize)
    gamma = [sparse(sum(gamma0[:,J], dims = 2)[:]) for J in basic_cells] # Only basic cell marginals remain
    return mu, nu, gamma
end

Random.seed!(0)

@testset ExtendedTestSet "solvecell!" begin
    # Solve cell first with cellsizes of size just one
    c = MOT.l22
    solver = DD.domdec_sinkhorn_stabilized!
    for N in [8, 13, 31]
        for cellsize in [1, 2, 3, 4]
            P = DomDecPlan(random_setup_domdec(N, cellsize)..., cellsize)
            ε = 2
            P.epsilon = ε
            params = (; solver_max_iter = 10000, 
                        solver_max_error = 1e-10, 
                        solver_max_error_rel=true, 
                        solver_verbose = true,
                        balance = true, 
                        truncate = true,
                        truncate_Ythresh = 1e-14, 
                        truncate_Ythresh_rel = true, 
                        epsilon = ε)
            
            k = 1
            j = 1
            P.partk = k
            DD.solvecell!(P, k, j, c, solver, params)

            # Check the primal-dual gap on the cell 
            gap = DD.cell_PD_gap(P, k, j, c, ε)
            @test abs(gap) < 1e-8 
        end
    end

    # TODO: tests in 2D and 3D

end

@testset ExtendedTestSet "iterate!" begin
    # Solve cell first with cellsizes of size just one
    c = MOT.l22
    solver = DD.domdec_sinkhorn_stabilized!
    for _iterate! in [DD.iterate_serial!, DD.iterate_parallel!]
        for N in [8, 13, 31]
            for cellsize in [1, 2, 3, 4]
                P = DomDecPlan(random_setup_domdec(N, cellsize)..., cellsize)
                ε = 2.
                P.epsilon = ε
                params = (; solver_max_iter = 10000, 
                            solver_max_error = 1e-10, 
                            solver_max_error_rel=true, 
                            solver_verbose = true,
                            balance = true, 
                            truncate = true,
                            truncate_Ythresh = 1e-14, 
                            truncate_Ythresh_rel = true, 
                            epsilon = ε)
                
                # Test two iterations independently
                for k in [1,2]
                    _iterate!(P, k, c, solver, params)
                    # Test that there's a solution on each cell
                    for j in eachindex(P.partitions[k])
                        # Check the primal-dual gap on the cell 
                        gap = DD.cell_PD_gap(P, k, j, c, ε)
                        @test abs(gap) < 1e-8 
                    end
                end

                # Last, perform a lot of iterations and check the primal-dual gap
                params = (; params..., 
                            domdec_iters = 8N,
                            parallel_iteration = false)
                P1 = deepcopy(P)
                DD.iterate!(P1, c, solver, params)

                gap = DD.PD_gap(P1, c, ε)
                @test gap < 1e-6

                # Do also for the parallel version
                params = (; params..., 
                            parallel_iteration = true)
                P1 = deepcopy(P)
                DD.iterate!(P1, c, solver, params)

                gap = DD.PD_gap(P1, c, ε)
                @test gap < 1e-6
            end
        end
    end
end

# TODO: tests for `plan_to_sparse_matrix`