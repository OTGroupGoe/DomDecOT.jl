
"""
    iterate!(P, c, solver, params::NamedTuple)

Perform `params[:iters]` inplace iterations of the domain decomposition algorithm 
on the plan `P`, using the cost `c` and the inner solver `solver`. 
If `params[:parallel_iteration]` is true, runs the iterations
in parallel.
"""
function iterate!(P::DomDecPlan, c, solver, params::NamedTuple)
    for k in 1:params.domdec_iters
        if params.parallel_iteration
            iterate_parallel!(P, k, c, solver, params)
        else
            iterate_serial!(P, k, c, solver, params)
        end
    end
end

"""
    iterate_serial!(P, k, c, solver, params)

Run an (inplace), serial half iteration `k` of domain decomposition 
on plan `P`, with cost `c`, solver `solver` and ceratain `params`.
"""
function iterate_serial!(P, k::Int, c, solver, params)
    k = (k-1)%length(P.partitions)+1
    P.partk = k
    P.epsilon = params[:epsilon]
    for j in eachindex(P.composite_cells[k])
        solvecell!(P, k, j, c, solver, params)
    end
end

"""
    iterate_parallel!(P, k, c, solver, params)

Run an (inplace), parallel half iteration `k` of domain decomposition 
on plan `P`, with cost `c`, solver `solver` and ceratain `params`.
"""
function iterate_parallel!(P, k::Int, c, solver, params)
    k = (k-1)%length(P.partitions)+1
    P.partk = k
    P.epsilon = params[:epsilon]
    Threads.@threads for j in eachindex(P.composite_cells[k])
        solvecell!(P, k, j, c, solver, params)
    end
end

"""
    solvecell!(P::DomDecPlan, k, j, c, solver, params)

Perform an (inplace) cell iteration on cell `P.partitions[k][j]` of the plan `P`,
using the inner `solver` and the given `params`.
"""
function solvecell!(P::DomDecPlan, k, j, c, solver, params)
    J = P.partitions[k][j]
    μJ = get_cell_X_marginal(P, k, j)
    all(μJ .> 0) || error("μJ must be strictly positive")
    νJ, I = get_cell_Y_marginal(P, k, j)
    XJ = view_X(P, J)
    YJ = view_Y(P, I)
    νI = view_Y_marginal(P, I)

    ε = params[:epsilon]
    K = [c(x, y) for y in eachcol(YJ), x in eachcol(XJ)]
    α = get_cell_alpha(P, k, j) 
    β = get_cell_beta(P, k, j)
    # β can have the wrong size after initialization or 
    # an alternated iteration. So we must be sure that 
    # it has the correct length:
    #fix_beta!(β, α, K, νJ, νI, μJ, ε) 
    # The version above was very inefficient; this one runs much faster
    fix_beta!(β, length(I))
    for i in eachindex(I) # TODO: wrap into function, make more efficient
        @inbounds β[i] = sum(K[i,ℓ] - α[ℓ] for ℓ in eachindex(J))/length(J)
    end
    # Solve the cell subproblem
    status = solver(β, α, νJ, νI, μJ, K, ε;
                    max_iter = params[:solver_max_iter],
                    max_error = params[:solver_max_error],
                    max_error_rel = params[:solver_max_error_rel],
                    verbose = params[:solver_verbose]
                    )

    if status == 2
        Core.print("warning: some iteration errored and returned an invalid plan.")
    end
    
    # Update cell parameters
    update_cell_plan!(P, K, I, k, j, μJ; 
                        balance = params[:balance],
                        truncate = params[:truncate],
                        truncate_Ythresh = params[:truncate_Ythresh],
                        truncate_Ythresh_rel = params[:truncate_Ythresh_rel]
                        )
    return status
end


###########################################################
###            Hierarchical iteration                   ###
###########################################################

function hierarchical_domdec(mu::AbstractMeasure, nu::AbstractMeasure, c, solver, params_schedule, layer0::Int = 3;
                            compute_PD_gap = false,
                            save_plans = false,
                            verbose = true)
    
    # Setup variables
    depth_mu = compute_multiscale_depth(mu)
    depth_nu = compute_multiscale_depth(nu)

    # Adjust to minimum depth
    depth = min(depth_mu, depth_nu)

    # Get multiscale measures
    muH = MultiScaleMeasure(mu; depth)
    nuH = MultiScaleMeasure(nu; depth)

    hierarchical_domdec(muH, nuH, c, solver, params_schedule, layer0;
                            compute_PD_gap,
                            save_plans,
                            verbose)
end

function hierarchical_domdec(muH::MultiScaleMeasure, nuH::MultiScaleMeasure, c, solver, params_schedule, layer0::Int = 3;
                            compute_PD_gap = false,
                            save_plans = false,
                            verbose = true)
    
    i = layer0
    depth = muH.depth

    # Measures at the coarsest level
    mu0 = muH[i]
    nu0 = nuH[i]

    # Find row of params for the first iteration
    k0 = findfirst(params_schedule.layer .== layer0)

    cellsize = params_schedule.cellsize[k0] # This actually shouldn't change in our implementation
    
    # Initial plan at the coarsest level
    π0 = sparse(nu0.weights .* mu0.weights')

    # Initialize initial DomDecPlan
    P = DomDecPlan(mu0, nu0, π0, cellsize)

    plans = typeof(P)[]

    # Keep track of elapsed times
    times = Dict(:solve => 0.0, :refine => 0.0)

    if verbose
        nthreads = (params_schedule.parallel_iteration[end] ? Threads.nthreads() : 1)
        Core.println("Running on ", nthreads, " threads\n")    

        if compute_PD_gap
            Core.println("Layer\tepsilon\ttime\tRelative PD gap")
        else
            Core.println("Layer\tepsilon\ttime")
        end
    end
    # Body of the iterations
    for k in k0:length(params_schedule)
        # Get parameters
        params = params_schedule[k]
        i = params.layer
    
        # Solve
        t0 = time()
        iterate!(P, c, solver, params)
        times[:solve] += time() - t0
        
        if verbose
            Core.print(i,"\t",params.epsilon,"\t",round(times[:solve]+times[:refine], digits = 2))
            if compute_PD_gap
                score1, score2 = DomDecOT.primal_and_dual_score(P, c, P.epsilon)
                Core.print("\t",(score1-score2)/score1 )
            end
            Core.println()
        end
        if save_plans
            push!(plans, (i, k, deepcopy(P)))
        end
        
        # Refine
        if (k < length(params_schedule)) && (i != params_schedule.layer[k+1])
            t0 = time()
            # global is needed when running this notebook as a script
            P = refine_plan(P, muH, nuH, i+1; consistency_check = false) 
            times[:refine] += time() - t0
            verbose && Core.println("   Refinement\t",round(times[:solve]+times[:refine], digits = 2),"\n")
        end
    end
    return P, plans, times 
end