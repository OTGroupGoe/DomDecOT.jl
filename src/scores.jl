"""
    cell_PD_gap(P, k, j, c, ε)

Primal-dual gap in cell j of partition k.
"""
function cell_PD_gap(P, k, j, c, ε)
    # Check the primal-dual gap on the cell 
    μJ = get_cell_X_marginal(P, k, j)
    νJ, I = get_cell_Y_marginal(P, k, j)
    νI = get_Y_marginal(P, I)
    XJ = get_cell_X(P, k, j)
    YJ = get_Y(P, I)
    PJ, _ = get_cell_plan(P, c, k, j) # This implicitly fixes beta
    aJ = get_cell_alpha(P, k, j)
    bJ = get_cell_beta(P, k, j) 
    bJ .+= ε.*(log.(νI) .- log.(νJ))
    MultiScaleOT.PD_gap_dense(bJ, aJ, PJ, c, YJ, XJ, νJ, μJ, ε)
end

"""
    primal_score(P, c, ε)

Compute the primal score of `P` for the entropic problem 
with cost function `c` and regularization `ε`, using only the non-zero
entries of `P`. 
"""
function primal_score(P, c, ε; balancing = false, truncation = 0)
    score = 0.0
    k = P.partk
    Part = P.partitions[k]
    for ℓ in eachindex(Part)
        K, I = get_cell_plan(P, c, k, ℓ, balancing, truncation) # TODO: Truncation is repeated
        C = get_cell_cost_matrix(P, c, k, ℓ, I)
        μJ = view_cell_X_marginal(P, k, ℓ)
        νI = view_Y_marginal(P, I)
        # Transport part
        score += dot(K, C)
        # Entropic part 
        score += ε*KL(K, νI .* μJ')
    end
    return score
end

"""
    dual_score(P, c,  ε)

Compute the dual score of `P` for the entropic problem 
with cost function `c` and regularization `ε`, using only the non-zero
entries of `P`. For the dense dual score, consider converting first
`P` to a sparse matrix and then using the routines of MultiScaleOT.
"""
function dual_score(P, c, ε; balancing = false, truncation = 0)
    k = P.partk
    Part = P.partitions[k]
    a, b = smooth_alpha_and_beta_fields(P, c)
    μ = P.mu.weights
    ν = P.nu.weights
    # Transport part
    score = dot(a, P.mu.weights) + dot(b, P.nu.weights)
    for ℓ in eachindex(Part)
        J = Part[ℓ]
        _, I = get_cell_plan(P, c, k, ℓ, balancing, truncation) # TODO: Truncation is repeated
        C = get_cell_cost_matrix(P, c, k, ℓ, I)
        # Entropic part
        for j in eachindex(J)
            for i in eachindex(I)
                score += ε*(1 - exp((a[J[j]]+b[I[i]]-C[i, j])/ε))*μ[J[j]]*ν[I[i]]
            end
        end
    end
    return score
end

"""
    PD_gap(P, c,  ε)

Primal-dual gap of plan `P` for the cost `c` and regularization `ε`.
"""
function PD_gap(P, c, ε)
    primal_score(P, c, ε) - dual_score(P, c, ε)
end