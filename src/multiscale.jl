"""
    refine_submeasure(v, u, u2, Js)

Refine the measure `v` with respect to the Y-cells given in `Js`
by matching the pattern induced by the refinement u -> u2.
"""
function refine_submeasure(v::SparseVector, u, u2, Js)
    r = copy(v)
    for i in r.nzind
        r[i] /= u[i]
    end
    # This doesn't work but something along the lines would be efficient
    #refined_index = [i for i in J for J in refinements[r.nzind]]
    # This in turn works but is a bit inneficient
    # TODO: delete entries with zero mass
    length(Js[r.nzind])>0 || return SparseVector(length(u2), Int[], Float64[])
    refined_index = vcat(Js[r.nzind]...)
    eltype(refined_index) == Int || error(string(refined_index))
    refined_values = zeros(Float64, length(refined_index))
    c = 1
    for i in r.nzind
        for j in Js[i]
            refined_values[c] = r[i]*u2[j]
            c+=1
        end
    end
    # Sort arrays for converting to sparse vector
    p = sortperm(refined_index)
    refined_index[p], refined_values[p]
end

"""
    refine_plan(P::DomDecPlan, 
                μH::MultiScaleMeasure{GridMeasure{D}}, 
                νH::MultiScaleMeasure, 
                k::Int;
                consistency_check = true) where D

Refine the plan P to match the refinement of the X marginal 
μH[k-1] -> μH[k] and of the Y marginal νH[k-1] -> νH[k]
"""
function refine_plan(P::DomDecPlan, 
                     μH::MultiScaleMeasure{GridMeasure{D}}, 
                     νH::MultiScaleMeasure, 
                     k::Int;
                     consistency_check = false) where D

    # Get new marginals and rest of data
    new_mu = μH[k]
    new_μ = new_mu.weights
    refinement_X = μH.refinements[k-1]
    new_nu = νH[k]
    new_ν = new_nu.weights
    n = length(new_ν)
    refinement_Y = νH.refinements[k-1]
    # TODO, LOW, FLEXIBILITY
    # Allow different cellsizes on each layer. But this would need of a different
    # refinement function, since current relies heavily on the fact that 
    # the number of new composite cells equals the number of old basic cells
    # (mathematically, `ceil(ceil(N/cellsize)/2) = ceil(ceil(N/2)/cellsize)` )
    cellsize = P.cellsize
    new_basic_cells, new_composite_cells = get_basic_and_composite_cells(new_mu.gridshape, cellsize)
    new_partitions = [get_partition(new_basic_cells, comp) for comp in new_composite_cells]
    new_gamma = Vector{SparseVector{Float64, Int64}}(undef, length(new_basic_cells))

    # When mu is a GridMeasure, there is the same number of A-composite cells
    # the in refined plan as of basic cells in coarse plan. 
    # This provides an easy way to implement the refinement of the paper.

    for i in eachindex(new_composite_cells[1])
        # Get refined measure coresponding to basic cell i of old plan
        νi = P.gamma[i]
        refined_index, refined_values = refine_submeasure(νi, P.nu.weights, new_ν, refinement_Y)
        new_νi = SparseVector{Float64, Int}(n, refined_index, refined_values)
        # Set this cell marginal to all basic cells in the
        # corresponding new composite cell
        mJ = sum(@views new_μ[new_partitions[1][i]])
        for j in new_composite_cells[1][i]
            # TODO, MEDIUM, PERFORMANCE: can we use the exact same sparse vector for
            # all these cell marginals (without copying)? If would reduce memory usage 
            # specially in the last layer, where most memory is used.
            new_gamma[j] = deepcopy(new_νi)
            new_gamma[j] .*= sum(@views new_μ[new_basic_cells[j]])/mJ
        end
    end

    # Copy the measures nu and mu, and update 
    # its entries with our new marginals (in case they are a bit different)
    new_nu_normalized = copy(new_nu)
    new_nu_normalized.weights = zeros(length(new_nu.weights))
    for νi in new_gamma
        new_nu_normalized.weights[νi.nzind] .+= νi.nzval
    end

    new_mu_normalized = copy(new_mu)
    new_mu_normalized.weights = deepcopy(new_mu.weights)
    for i in eachindex(new_basic_cells)
        mi = sum(@views new_mu.weights[new_basic_cells[i]])
        new_mu_normalized.weights[new_basic_cells[i]] .*= sum(new_gamma[i])/mi
    end

    # Get duals
    # Get current alpha field and refine it
    alpha_field = smooth_alpha_field(P)
    nodes = MultiScaleOT.get_grid_nodes(P.mu.points, P.mu.gridshape)
    #new_nodes = get_grid_nodes(newSP.X, newSP.shapeX)
    new_alpha_field = MultiScaleOT.refine_dual(alpha_field, nodes, new_mu.points, P.mu.gridshape)

    # Get new cell alphas
    new_alphas = [[new_alpha_field[J] for J in part] for part in new_partitions]
    
    # We will left the betas uninitialized; in the first iteration
    # with the new plan they will be fixed
    new_betas = [[Float64[] for _ in part] for part in new_partitions]

    # If sufficiently tested turn off the consistency check
    return DomDecPlan(new_mu_normalized, 
                    new_nu_normalized::AbstractMeasure, 
                    new_gamma,
                    cellsize::Int, 
                    new_basic_cells::Vector, 
                    new_composite_cells::Vector, 
                    new_partitions::Vector,
                    new_alphas::Vector, 
                    new_betas::Vector, 
                    P.epsilon, 
                    0; 
                    consistency_check)
end
