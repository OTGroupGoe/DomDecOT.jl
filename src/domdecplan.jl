# CODE STATUS: PARTIALLY REVISED, PARTIALLY TESTED
using SparseArrays
import MultiScaleOT as MOT 
import MultiScaleOT: AbstractMeasure, GridMeasure, npoints

"""
    AbstractPlan

Super type of all implementations of OT plans.
"""
abstract type AbstractPlan{D} end

# This should inherit from AbstractPlan, but we are still to find 
# the appropriate type signature to make it as well type stable.
mutable struct DomDecPlan{M<:AbstractMeasure, N<:AbstractMeasure}
    mu::M # X measure 
    nu::N # Y measure 
    gamma::Vector{SparseVector{Float64, Int64}} # Y-marginals on basic cells
    cellsize::Int
    basic_cells::Vector{Vector{Int}} # basic cells 
    composite_cells::Vector{Vector{Vector{Int}}} # composite cells 
    partitions::Vector{Vector{Vector{Int}}} # partitions 
    alphas::Vector{Vector{Vector{Float64}}} # X potentials 
    betas::Vector{Vector{Vector{Float64}}}  # Y potentials
    epsilon::Float64 # epsilon for which alphas, betas are dual cell potentials
    partk::Int     # Current iterate
    

    """
        DomDecPlan(mu::AbstractMeasure{D}, nu::AbstractMeasure, gamma,
                    cellsize::Int[, basic_cells::Vector, 
                    composite_cells::Vector, partitions::Vector,
                    alphas::Vector, betas::Vector, 
                    epsilon=1.0, partk=1; 
                    consistency_check = true])

    A DomDecPlan is a struct that keeps track of the status of the 
    domain decomposition algorithm in an effient manner. Its arguments are

    * `mu`: AbstractMeasure representing the X marginal
    * `nu`: AbstractMeasure representing the Y marginal
    * `gamma`: `gamma[i]` is a sparse vector representing the current marginal
      of basic cell `i`.
    * `cellsize::Int`: maximum size of the basic cells (along all dimensions).
    * `basic_cells`: `basic_cells[i]` is the indices of the atoms in `mu` that
      are merged together to form a basic cell. 
    * `composite_cells`: `composite_cells[k][j]` refers to the group of basic cells
      that constitute the `j`-th subdomain of the `k`-th partition.
    * `partitions`: `partitions[k][j]` are the indices of all the `X` atoms that constitute the 
      first marginal during `j`-th subdomain of partition `k`. It equals 
      `vcat([basic_cells[composite_cells[k][j]]...)`
    * `alphas`: X-dual potential on each subdomain. `alphas[k][j]` has the same length as `partitions[k][j]`.
    * `betas`: Y-dual potentials on each subdomain.
    * `epsilon`: last global epsilon used to solve the cell problems.
    * `partk`: index of the last partition whose subdomains were solved.
    """
    function DomDecPlan(mu::AbstractMeasure{D}, 
                        nu::AbstractMeasure, 
                        gamma,
                        cellsize::Int, 
                        basic_cells::Vector, 
                        composite_cells::Vector, 
                        partitions::Vector,
                        alphas::Vector, 
                        betas::Vector, 
                        epsilon=1.0, 
                        partk=1; 
                        consistency_check = true) where D
        if consistency_check
            # We assume that mu, nu are consistent measures
            all([all(1 .≤ cell .≤ length(mu.weights)) for cell in basic_cells]) || error("basic cells not compatible with number of atoms in mu")
            all(length.(gamma) .== length(nu.weights)) || error("size of second marginal of gamma doesn't match nu")
            all([sum(mu.weights[J]) .≈ sum(gamma[i]) for (i,J) in enumerate(basic_cells)])  || error("first marginal of gamma does not match mu")
            all(nu.weights .≈ sum(gamma))  || error("second marginal of gamma does not match nu")
            all(length.(basic_cells) .≤ cellsize^D) || error("some basic cell is bigger than cellsize^D")
            length(composite_cells) == length(partitions) || error("number of composite cell lists does not match number of partitions")
            for i in eachindex(composite_cells)
                get_partition(basic_cells, composite_cells[i]) == partitions[i] || error("cells not compatible with partitions")
            end
            [length.(alpha) for alpha in alphas] == [length.(part) for part in partitions] || error("potentials not compatible with partitions")
            length.(alphas) == length.(betas) || error("some partition have different number of alphas and betas")
        end

        new{typeof(mu), typeof(nu)}(mu, nu, gamma, 
                                    cellsize, basic_cells, 
                                    composite_cells, partitions,
                                    alphas, betas, epsilon, partk)
    end
end

function DomDecPlan(mu::AbstractMeasure, nu::AbstractMeasure, gamma,
            cellsize, basic_cells::Vector, 
            composite_cells::Vector, partitions::Vector,
            epsilon=1.0, partk=1; consistency_check = true) 
    
    alphas = [[zeros(length(J)) for J in part] for part in partitions]
    betas = [[Float64[] for _ in part] for part in partitions]
    return DomDecPlan(mu, nu, gamma, 
                    cellsize, basic_cells, 
                    composite_cells, partitions,
                    alphas, betas, 
                    epsilon, partk; consistency_check)     
end

function DomDecPlan(mu::GridMeasure{D}, nu::AbstractMeasure, gamma, cellsize::Int, epsilon::Float64=1.0, partk=1; consistency_check = true) where D
    # If mu is a GridMesaure, basic and composite cells are straightforward to Obtain
    # TODO: how to handle CloudMeasure
    basic_cells, composite_cells = get_basic_and_composite_cells(mu.gridshape, cellsize)
    partitions = [get_partition(basic_cells, comp) for comp in composite_cells]
    return DomDecPlan(mu, nu, gamma, 
                    cellsize, basic_cells, 
                    composite_cells, partitions,
                    epsilon, partk; consistency_check)
end

###############################################################################
# Elementary operations to extract information about the cells, 
# like their X, weights, and so on.
###############################################################################

# TODO: docstrings

get_X_massive_points(P::DomDecPlan, J) = J[P.mu.weights[J] .> 0]

get_cell_X_massive_points(P::DomDecPlan, k, j) = get_X_massive_points(P, P.partitions[k][j])

get_X_marginal(P::DomDecPlan, J) = P.mu.weights[J]

get_cell_X_marginal(P::DomDecPlan, k, j) = get_X_marginal(P, P.partitions[k][j])

view_X_marginal(P::DomDecPlan, J) = @views P.mu.weights[J]

view_cell_X_marginal(P::DomDecPlan, k, j) = view_X_marginal(P, P.partitions[k][j])

"""
    get_cell_Y_marginal(P::DomDecPlan, J)

Compute the total Y-marginal of a set of basic cells `J`
"""
function get_cell_Y_marginal(P::DomDecPlan, J)
    # TODO, MEDIUM, PERFORMANCE
    # Check if this is performant enough
    νJ = sum(P.gamma[J])
    return νJ.nzval, νJ.nzind
end

get_cell_Y_marginal(P::DomDecPlan, k, j) = get_cell_Y_marginal(P, P.composite_cells[k][j])

get_Y_marginal(P::DomDecPlan, I) = P.nu.weights[I]

view_Y_marginal(P::DomDecPlan, I) = @views P.nu.weights[I]

"""
    view_X(P, J)

Return a view of the X points in subset `J` of `P.mu.points`.
"""
view_X(P::DomDecPlan, J) = @views P.mu.points[:,J]

view_cell_X(P::DomDecPlan, k, j) = view_X(P, P.partitions[k][j])

get_X(P::DomDecPlan, J) = P.mu.points[:,J]

get_cell_X(P::DomDecPlan, k, j) = get_X(P, P.partitions[k][j])

"""
    view_cell_Y(P, I)

Return a view of the Y points with index `I`.
"""
view_Y(P::DomDecPlan, I) = @views P.nu.points[:,I]

get_Y(P::DomDecPlan, I) = P.nu.points[:,I]

function view_cell_Y(P, k, j)
    _, I = get_cell_Y_marginal(P::DomDecPlan, k, j)
    return view_Y(P, I)
end

function get_cell_Y(P, k, j)
    _, I = get_cell_Y_marginal(P::DomDecPlan, k, j)
    return get_Y(P, I)
end

get_cell_alpha(P::DomDecPlan, k, j) = P.alphas[k][j]

get_cell_beta(P::DomDecPlan, k, j) = P.betas[k][j]

# Another option could be to use the value of alpha to compute beta
# function get_cell_beta(P::DomDecPlan, k, j) 
#     C = get_cell_cost_matrix(P, k, j, I)
#     C .-= eps.*log.(νJ) 
#     C .-= eps.*log.(μJ')
#     Cα = logsumexp(C, α, eps, 1)
#     β = eps.*log.(νJ) .+ Cα
# end

function get_cell_cost_matrix(P::DomDecPlan, c, J, I)
    XJ = view_X(P, J)
    YJ = view_Y(P, I)
    C = [c(x, y) for y in eachcol(YJ), x in eachcol(XJ)]
    return C
end

get_cell_cost_matrix(P::DomDecPlan, c, k, j, I) = get_cell_cost_matrix(P, c, P.partitions[k][j], I)

function get_cell_plan(P::DomDecPlan, c, k, j, balancing = true, truncation = 1e-14) 
    μJ = get_cell_X_marginal(P, k, j)
    νJ, I = get_cell_Y_marginal(P, k, j)
    # If the following line was commented the code would still work
    # but the βs would be more difficult to glue (they wouldn't share the same Y-marginal)
    νI = view_Y_marginal(P, I)
    α = get_cell_alpha(P, k, j)
    β = get_cell_beta(P, k, j)
    C = get_cell_cost_matrix(P, c, k, j, I)
    fix_beta!(β, α, C, νJ, νI, μJ, P.epsilon)
    ε = P.epsilon
    # Rename C for code clarity
    K = C
    get_kernel!(K, β, α, νI, μJ, ε)
    # Maybe make these balancing and truncation more flexible
    if balancing
        balance!(K, μJ, 1e-14)
    end
    if truncation>0
        truncate!(K, truncation)
    end
    # Return the plan and where it lives
    return K, I
end

function reduce_cellplan_to_cellmarginals(P::DomDecPlan, PJ::AbstractMatrix, k, j, μJ) where {D}
    J = P.composite_cells[k][j]
    Pis = zeros(size(PJ, 1), length(J))
    mJ = zeros(length(J)) # Mass of each basic cell
    ptr = 0
    for i in eachindex(J)
        cellsize = length(P.basic_cells[J[i]])
        for m in 1:cellsize
            Pis[:, i] .+= @views PJ[:,ptr+m]
        end
        mJ[i] = sum(@views μJ[ptr+1:ptr+cellsize])
        ptr += cellsize
    end
    return Pis, mJ
end

###############################################################################
# Truncation routines 
###############################################################################

"""
    truncate!(ν::AbstractArray, threshold)

Drop values of `ν` smaller or equal than `threshold`. Acts inplace.
"""
function truncate!(v::AbstractArray, threshold)
    v[v .< threshold] .= 0
end

# Relative threshold version
function truncate!(A::AbstractMatrix, μJ, rel_threshold)
    for i in 1:size(A,2)
        truncate!(view(A, :, i), μJ[i]*rel_threshold)
    end
end

truncate!(v::SparseVector, threshold) = droptol!(v, threshold)

"""
    truncate!(P::AbstractPlan, rel_threshold)

Drop values of the plan `P` that are smaller than `rel_threshold`
with respect to the cell mass. Acts inplace.
"""
function truncate!(P::DomDecPlan, rel_threshold)
    for (i,J) in enumerate(P.basic_cells)
        truncate!(P.gamma[i], rel_threshold*sum(P.mu.weights[J]))
    end
end

###############################################################################
# Routines for updating the variables of a given composite cell 
# when the cell problem has been solved
###############################################################################

# TODO: update following docstring
"""
    update_cell_plan!(P::DomDecPlan, I, J, Psub)

Update cell `J` of the plan `P`. The rest of arguments are:

* `I`: Y indices with positive mass in the composite cell.
* `Psub`: Plan in cell `J` and Y indices `I`.
"""
function update_cell_plan!(P::DomDecPlan, P_basic, k, j0, I)
    basic_cells = P.composite_cells[k][j0]
    for i in eachindex(basic_cells)
        # Construct sparse vectors only with the positive entries
        Ip = findall(>(0), @views P_basic[:,i])
        P.gamma[basic_cells[i]] = sparsevec(I[Ip], P_basic[Ip,i], npoints(P.nu))
    end
end

function update_cell_plan!(P::DomDecPlan, PJ, I, k, j, μJ;
        balance = true, truncate = true,
        truncate_Ythresh = 1e-16, truncate_Ythresh_rel=true)
    
    # Compute target cell masses
    P_basic, mJ = reduce_cellplan_to_cellmarginals(P, PJ, k, j, μJ)

    if balance
        balance!(P_basic, mJ, truncate_Ythresh)
    end
    if truncate
        truncate!(P_basic, mJ, truncate_Ythresh)
    end

    update_cell_plan!(P, P_basic, k, j, I)
    # TODO: efficient way, instead of creating sparsevec in each iteration?
    nothing
end

#######################################################
# Utils for transforming a plan to dense/sparse matrices
# and viceversa.
#######################################################

"""
    reduce_to_cellsize(γ0::AbstractVector, basic_cells)

Add together marginals corresponding to the same basic cell.
"""
function reduce_to_cells(γ0::AbstractVector{AbstractVector}, basic_cells)
    gamma = [sum(@views γ0[cell]) for cell in basic_cells]
    return gamma
end

# This is probably the only use case of reduce_to_cellsize: from sparsematrix/matrix to domdecplan
# TODO: check if the following is used somewhere
# TODO: would it be better someething like "sparse_matrix_to_domdecplan"?
# Or just DomDecPlan(sparse_matrix, cell_size)
function reduce_to_cells(K::AbstractMatrix, basic_cells)
    gamma = SparseVector{Float64, Int}[]
    for cell in basic_cells
        v = @views sum(K[:, cell], dims=2)
        push!(gamma, sparsevec(v))
    end
    return gamma
end

# TODO: Test this in iterate, solving with sinkhorn and with domdec
"""
    reduce_to_cellsize(gamma, gridshape, cellSize)

Add together marginals corresponding to the same basic cell, 
for given `shapeX` and `cellSize`.
"""
function reduce_to_cells(gamma, gridshape, cellsize)
    basic_cells, _ = get_cells(gridshape, cellsize)
    reduce_to_cellsize(gamma, basic_cells)
end

function DomDecPlan(mu::GridMeasure{D}, nu::AbstractMeasure, A::AbstractMatrix, 
                    cellsize::Int = 2, epsilon::Float64 = 1.0; consistency_check = true) where D
    
    gridshape = mu.gridshape
    basic_cells, composite_cells = get_basic_and_composite_cells(gridshape, cellsize)
    partitions = [get_partition(basic_cells, comp) for comp in composite_cells]
    gamma = reduce_to_cells(A, basic_cells)
    return DomDecPlan(mu, nu, gamma, 
                    cellsize, basic_cells, 
                    composite_cells, partitions,
                    epsilon; consistency_check)
end
    
# This returns a dense matrix for the moment
function plan_to_dense_matrix(P::DomDecPlan, c, k, balancing = true)
    k = (k-1)%length(P.partitions)+1
    Part = P.partitions[k]
    A = zeros(npoints(P.nu), npoints(P.mu))
    for j in 1:length(Part)
        K, I = get_cell_plan(P, c, k, j, balancing)
        A[I, Part[j]] .= K
    end
    return A
end

# TODO: Same here, test this in iterate
# Maybe call just SparseMatrix(::DomDecPlan)
function plan_to_sparse_matrix(P::DomDecPlan, c, balancing = true, truncate_Ythresh = 0)
    # This follows very closely MultiScaleOT.refine_support
    k = P.partk
    Part = P.partitions[k]
    N = npoints(P.nu)
    M = npoints(P.mu)
    nzval_scattered = Vector{Vector{Float64}}(undef, N)
    rowval_scattered = Vector{Vector{Int}}(undef, N) # Entry `i` is the support of column `i`
    colptr = Vector{Int}(undef, N+1) # Entry `i+1` is the length of col `i`; a cumsum then yields the correct colptr.
    colptr[1] = 1
    for ℓ in eachindex(Part)
        K, I = get_cell_plan(P, c, k, ℓ, balancing, truncate_Ythresh) # TODO: Truncation is repeated
        for j in eachindex(Part[ℓ])
            i = Part[ℓ][j]
            # j denotes column of K corresponding to column Part[j] in sparse array
            # Find indices of positive entries
            Ip = findall(>(truncate_Ythresh), @views K[:,j])
            rowval_scattered[i] = I[Ip]
            nzval_scattered[i] = K[Ip,j]
            colptr[i+1] = length(Ip)
        end
    end
    # Concatenate all columns
    rowval = vcat(rowval_scattered...)
    nzval = vcat(nzval_scattered...)
    # Do a cumsum inplace to get the correct colptr
    cumsum!(colptr, colptr)
    # Construct the sparse matrix
    return SparseMatrixCSC{Float64, Int}(N, M, colptr, rowval, nzval)
end