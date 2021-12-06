# CODE STATUS: TESTS NOT DOCUMENTED
using SparseArrays
import MultiScaleOT as MOT 
import MultiScaleOT: AbstractMeasure, GridMeasure, npoints, normalize!
import DomDecOT as DD
import DomDecOT: DomDecPlan
import Random
import StatsBase: mean

function random_setup_domdec(N, cellsize)
    x1 = collect(1.:N)
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

@testset ExtendedTestSet "hierarchical_domdec" begin
    # Solve cell first with cellsizes of size just one
    c = MOT.l22
    solver = DD.domdec_sinkhorn_stabilized!
    for parallel_iteration in [true, false]
        for N in [64, 63] # Try with size that is not 2^n
            mu, nu, _ = random_setup_domdec(N, 1)
            for cellsize in [1, 2, 3, 4]
                muH = MOT.MultiScaleMeasure(mu)
                nuH = MOT.MultiScaleMeasure(nu)

                # +
                depth = muH.depth
                        
                # Epsilon schedule
                Nsteps = 3
                factor = 2.
                eps_target = 0.5
                last_iter = fill(eps_target/2, 4)

                layer_schedule, eps_schedule, iters_schedule = DD.default_domdec_eps_schedule(depth, eps_target; 
                                                                                        Nsteps, factor, last_iter)
                parallel_iteration = false

                params_schedule = DD.make_domdec_schedule(
                                    layer = layer_schedule,
                                    epsilon = eps_schedule, 
                                    solver_max_error = 1e-6,
                                    solver_max_error_rel=true, 
                                    solver_max_iter = 10000, 
                                    solver_verbose = true,
                                    balance = true,
                                    truncate = true,
                                    truncate_Ythresh = 1e-15, 
                                    truncate_Ythresh_rel = false, 
                                    parallel_iteration = parallel_iteration,
                                    domdec_iters = iters_schedule,
                                    cellsize = cellsize
                            );

                c(x,y) = MOT.l22(x,y)
                P, _, _ = DD.hierarchical_domdec(muH, nuH, c, solver, params_schedule, 2;
                                                        compute_PD_gap = false,
                                                        save_plans = false,
                                                        verbose = false)

                ε = eps_target/2
                K = DD.plan_to_dense_matrix(P, c)
                a, b = DD.smooth_alpha_and_beta_fields(P, c)
                
                score1 = MOT.primal_score_dense(K, c, nu, mu, ε)
                score2 = MOT.dual_score_dense(b, a, c, nu, mu, ε)

                @test (score1-score2)/score1 < 1e-5
                @test MOT.l1(sum(K, dims=1), mu.weights)<1e-8
                @test MOT.l1(sum(K, dims=2), nu.weights)<1e-8
            end
        end
    end
end

# TODO: tests for `plan_to_sparse_matrix`