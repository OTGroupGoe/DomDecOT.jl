import MultiScaleOT: KL
import LinearAlgebra: mul!

"""
    fix_beta!(β, N::Int)

Check if the Y-dual parameter has the appropriate size. If not, turns it 
into a zeros(N). 
Return `true` if β was of the appropriate length, `false` otherwise
"""
function fix_beta!(β, N::Int)
    if length(β) != N
        β .= 0
        if length(β) < N
            sizehint!(β, N)
            for _ in length(β)+1:N; push!(β, 0); end
        else
            for _ in length(β):-1:N+1; pop!(β); end
        end
        return false
    end
    return true
end

function fix_beta!(β, α, K, νJ, νI, μJ, ε, K_is_cost = false)
    M, N = size(K)
    was_alright = fix_beta!(β, M)
    if !was_alright
        if K_is_cost
            get_kernel!(K, β, α, νI, μJ, ε)
        end
        mul!(β, K, ones(N))
        # Now we do β .= νJ ./ β; β .= ε .* log.(β)
        # Which can be sumarized as
        β .= ε .* log.(νJ ./ β)
    end
    return was_alright
end

"""
    get_kernel!(C, a, b, μ, ν, eps)

Compute inplace the Gibbs energy of matrix `C`, current duals `a` and `b` and scale it with marginals `μ`
and `ν`.
"""
function get_kernel!(C, a, b, μ, ν, eps)
    C .= μ .* exp.((a .+ b' .- C)./eps) .* ν'
    nothing
end

get_kernel!(C, a, b, eps) = get_kernel!(C, a, b, 1, 1, eps)

"""
    get_kernel!(K, μ, ν, eps)

Compute the Gibbs energy of matrix `C`, current duals `a` and `b` and scale it with marginals `μ`
and `ν`.
"""
function get_kernel(C, a, b, μ, ν, eps) 
    K = copy(C)
    get_kernel!(K, a, b, μ, ν, eps)
    return K
end

get_kernel(C, a, b, eps) = get_kernel(C, a, b, 1, 1, eps)

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
    MOT.PD_gap_dense(bJ, aJ, PJ, c, YJ, XJ, νJ, μJ, ε)
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