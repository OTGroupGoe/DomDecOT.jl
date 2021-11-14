# CODE STATUS: PARTIALLY REVISED, NOT TESTED

###########################################
# DomDecPlan
##############################

function iterate!(P, c, _iterate!, solver, params)

    Npart = length(P.composite_cells)
    P.epsilon = params[:epsilon]
    for k in params[:iters]
        k0 = (k-1)%Npart+1
        P.partk = k0
        _iterate!(P, k0, c, solver, params)
    end
end

"""
    iterate!(P, c, eps, k, composite_partitions, solver_params, plan_params)

Run an (inplace) half iteration `k` of the domain decomposition algorithm for entropic
optimal transport (https://arxiv.org/abs/2001.10986) on plan `P`, with cost `c`,
regularization `eps`, on the `composite_partitions` and with a certain
`solver`.
"""
function iterate_serial!(P, k, c, solver, params)
    for j in eachindex(P.composite_cells[k])
        solvecell!(P, k, j, c, solver, params)
    end
end

function iterate_parallel!(P, k, c, solver, params)
    Threads.@threads for j in eachindex(P.composite_cells[k])
        solvecell!(P, k, j, c, solver, params)
    end
end

"""
    solvecell!(P::DomDecPlan, k, j, c, solver, params)

Perform an (inplace) cell iteration on cell `P.partitions[k][j]` of the plan `P`.
"""
function solvecell!(P::DomDecPlan, k, j, c, solver, params)
    J = P.partitions[k][j]
    μJ = get_cell_X_marginal(P, k, j)
    all(μJ .> 0) || error("μJ must be strictly positive")
    νJ, I = get_cell_Y_marginal(P, k, j)
    XJ = view_X(P, J)
    YJ = view_Y(P, I)
    νI = view_Y_marginal(P, I)

    K = [c(x, y) for y in eachcol(YJ), x in eachcol(XJ)]
    α = get_cell_alpha(P, k, j) 
    β = get_cell_beta(P, k, j)
    # β can have the wrong size after initialization or 
    # an alternated iteration. So we must be sure that 
    # it has the correct length
    fix_beta!(β, length(I))

    # Solve the cell subproblem
    status = solver(β, α, νJ, νI, μJ, K, params[:epsilon];
                    max_iter = params[:solver_max_iter],
                    max_error = params[:solver_max_error],
                    max_error_rel = params[:solver_max_error_rel],
                    verbose = params[:solver_verbose]
                    )
    
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