
"""
    iterate!(P, c, solver, params)

Perform `params[:iters]` inplace iterations of the domain decomposition algorithm 
on the plan `P`, using the cost `c` and the inner solver `solver`. 
If `params[:parallel_iteration]` is true, runs the iterations
in parallel.
"""
function iterate!(P, c, solver, params)
    for k in params[:iters]
        if params[:parallel_iteration]
            iterate_parallel!(P, k, c, solver, params)
        else
            iterate_serial!(P, k, c, solver, params)
        end
    end
end

"""
    iterate_serial!(P, k, c, solver, params)

Run an (inplace), serial half iteration `k` of domain decomposition 
on plan `P`, with cost `c`, solver `solver` and ceratin `params`.
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
    iterate!(P, k, c, solver, params)

Run an (inplace), parallel half iteration `k` of domain decomposition 
on plan `P`, with cost `c`, solver `solver` and ceratin `params`.
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
    fix_beta!(β, α, K, νJ, νI, μJ, ε) 
    # Solve the cell subproblem
    status = solver(β, α, νJ, νI, μJ, K, ε;
                    max_iter = params[:solver_max_iter],
                    max_error = params[:solver_max_error],
                    max_error_rel = params[:solver_max_error_rel],
                    verbose = params[:solver_verbose]
                    )

    if status == 2
        print("warning: some iteration errored and returned an invalid plan.")
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

# TODO: hierarchical iteration, following the same scheme
# as the MOT library