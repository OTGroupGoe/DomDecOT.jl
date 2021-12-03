
"""
    AbstractPlan

Super type of all implementations of OT plans.
"""
abstract type AbstractPlan end

# TODO: This should inherit from AbstractPlan, but we are still to find 
# the appropriate type signature to make it as well type stable.
"""
    DomDecPlan(mu::AbstractMeasure, nu::AbstractMeasure, gamma,
                cellsize::Int[, basic_cells::Vector, 
                composite_cells::Vector, partitions::Vector,
                alphas::Vector, betas::Vector, 
                epsilon::Float64, partk::Int)

A DomDecPlan is a struct that keeps track of the status of the 
domain decomposition algorithm in an effient manner. Its arguments are

* `mu`: AbstractMeasure representing the X marginal
* `nu`: AbstractMeasure representing the Y marginal
* `gamma`: `gamma[i]` is a sparse vector representing the current marginal
    of basic cell `i`. Alternatively, `gamma` can also be a sparse matrix
    representing the full initial plan.
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
mutable struct DomDecPlan{M<:AbstractMeasure, N<:AbstractMeasure} <: AbstractPlan
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
        DomDecPlan(mu::AbstractMeasure{D}, 
                    nu::AbstractMeasure, 
                    gamma,
                    cellsize::Int 
                    [,
                    basic_cells::Vector, 
                    composite_cells::Vector, 
                    partitions::Vector,
                    alphas::Vector, 
                    betas::Vector
                    epsilon=1.0, 
                    partk=1; 
                    consistency_check = true
                    ]) where D
    
    Build a DomDecPlan from the given arguments. Usual calls are

    ```julia 
    DomDecPlan(mu, nu, gamma, cellsize)
    ```
    if there are no precomputed cells, or 
    ```julia 
    DomDecPlan(mu, nu, gamma, cellsize, basic_cells, composite_cells, partitions)
    ```
    if there are already computed, or even
    ```julia 
    DomDecPlan(mu, nu, gamma, cellsize, basic_cells, composite_cells, partitions,
               alphas, betas)
    ```  
    if all parameters are already computed (for example, after a refinement step).
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
                        partk=0; 
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
            epsilon=1.0, partk=0; consistency_check = true) 
    
    alphas = [[zeros(length(J)) for J in part] for part in partitions]
    betas = [[Float64[] for _ in part] for part in partitions]
    return DomDecPlan(mu, nu, gamma, 
                    cellsize, basic_cells, 
                    composite_cells, partitions,
                    alphas, betas, 
                    epsilon, partk; consistency_check)     
end

function DomDecPlan(mu::GridMeasure{D}, nu::AbstractMeasure, gamma, 
                    cellsize::Int, epsilon::Float64=1.0, partk=0; 
                    consistency_check = true) where D
    # If mu is a GridMesaure, basic and composite cells are straightforward to Obtain
    # TODO: how to handle CloudMeasure
    basic_cells, composite_cells = get_basic_and_composite_cells(mu.gridshape, cellsize)
    partitions = [get_partition(basic_cells, comp) for comp in composite_cells]
    return DomDecPlan(mu, nu, gamma, 
                    cellsize, basic_cells, 
                    composite_cells, partitions,
                    epsilon, partk; consistency_check)
end

function DomDecPlan(mu::CloudMeasure, nu::AbstractMeasure, gamma, cellsize, partk=0, epsilon::Float64=1.0; consistency_check = true) where D
    error("not implemented for mu::CloudMeasure (but feel free to open a PR!)")
end

function DomDecPlan(mu::AbstractMeasure, nu::AbstractMeasure, gamma, cellsize, partk=0, epsilon::Float64=1.0; consistency_check = true) where D
    error("not implemented for mu a general AbstractMeasure")
end

###############################################################################
# Elementary operations to extract information about the cells, 
# like their X, weights, and so on.
# Many functions have two versions: one called `get_foo(P, J)`, which gets
# some property of P in some spatial region J, and another `get_cell_foo(P, k, j)`,
# which evaluates to the corresponding `get_foo(P, J)` where `J` is the relevant 
# spatial set, obtained from the cell `j` of partition `k`.
# We leave the type of P unasserted as long as we can, so that
# future plans can reuse these functions as much as possible.
###############################################################################

# TODO: Some docstrings missing, but it should be merely on those unexported and
# whose functioning is obvious.

get_X_massive_points(P, J) = J[P.mu.weights[J] .> 0]

get_cell_X_massive_points(P, k, j) = get_X_massive_points(P, P.partitions[k][j])

get_X_marginal(P, J) = P.mu.weights[J]

get_cell_X_marginal(P, k, j) = get_X_marginal(P, P.partitions[k][j])

view_X_marginal(P, J) = @views P.mu.weights[J]

view_cell_X_marginal(P, k, j) = view_X_marginal(P, P.partitions[k][j])

"""
    get_cell_Y_marginal(P, J)

Compute the total Y-marginal of a set of **basic cells** `J`
"""
function get_cell_Y_marginal(P::DomDecPlan, J)
    # TODO, MEDIUM, PERFORMANCE
    # Check if this is performant enough
    νJ = sum(P.gamma[J])
    return νJ.nzval, copy(νJ.nzind)
end

get_cell_Y_marginal(P::DomDecPlan, k, j) = get_cell_Y_marginal(P, P.composite_cells[k][j])

get_Y_marginal(P, I) = P.nu.weights[I]

view_Y_marginal(P, I) = @views P.nu.weights[I]

"""
    view_X(P, J)

Return a view of the X points in subset `J` of `P.mu.points`.
"""
view_X(P, J) = @views P.mu.points[:,J]

view_cell_X(P, k, j) = view_X(P, P.partitions[k][j])

get_X(P, J) = P.mu.points[:,J]

get_cell_X(P, k, j) = get_X(P, P.partitions[k][j])

"""
    view_cell_Y(P, I)

Return a view of the Y points with index `I`.
"""
view_Y(P, I) = @views P.nu.points[:,I]

get_Y(P, I) = P.nu.points[:,I]

function view_cell_Y(P, k, j)
    _, I = get_cell_Y_marginal(P, k, j)
    return view_Y(P, I)
end

function get_cell_Y(P, k, j)
    _, I = get_cell_Y_marginal(P, k, j)
    return get_Y(P, I)
end

get_cell_alpha(P, k, j) = P.alphas[k][j]

get_cell_beta(P, k, j) = P.betas[k][j]

# TODO: Another option could be to use the value of alpha to compute beta
# and remove all dependences on a `betas` parameter. To see how check
# function `fix_beta!` in src/duals.jl.

"""
    get_cost_matrix(P, c, J, I)

Return the cost matrix corresponding to points Y[:,I] and X[:,J].
"""
function get_cost_matrix(P, c, J, I)
    XJ = view_X(P, J)
    YJ = view_Y(P, I)
    C = [c(x, y) for y in eachcol(YJ), x in eachcol(XJ)]
    return C
end

get_cell_cost_matrix(P, c, k, j, I) = get_cost_matrix(P, c, P.partitions[k][j], I)

function get_cell_plan(P, c, k, j, balancing = true, truncation = 1e-14) 
    # Get cell paramteres
    μJ = get_cell_X_marginal(P, k, j)
    νJ, I = get_cell_Y_marginal(P, k, j)
    if P.partk ≤ 0 # no iterations happened, return product plan
        return νJ.*μJ'./sum(μJ), I
    end # else:
    νI = view_Y_marginal(P, I)
    α = get_cell_alpha(P, k, j)
    β = get_cell_beta(P, k, j)
    C = get_cell_cost_matrix(P, c, k, j, I)

    ε = P.epsilon
    fix_beta!(β, α, C, νJ, νI, μJ, P.epsilon)
    # Rename C for code clarity
    K = C
    # K is scaled on Y with the *global* Y marginal
    get_kernel!(K, β, α, νI, μJ, ε)
    # Maybe make these balancing and truncation more flexible
    if balancing
        balance!(K, μJ, 1e-14)
    end
    if truncation>0
        truncate!(K, truncation)
    end
    # Return the plan and its Y support
    return K, I
end

"""
    reduce_cellplan_to_cellmarginals(P::DomDecPlan, PJ::AbstractMatrix, k, j, μJ) 

Take the matrix `PJ`, that is assumed to be the current plan on cell `k` of partition `k`
and extract its cell marginals.
"""
function reduce_cellplan_to_cellmarginals(P::DomDecPlan, PJ::AbstractMatrix, k, j, μJ) 
    J = P.composite_cells[k][j]
    Pis = zeros(size(PJ, 1), length(J))
    mJ = zeros(length(J)) # Mass of each basic cell
    ptr = 0
    for i in eachindex(J)
        cellsize = length(P.basic_cells[J[i]])
        # To be completely sure that the inbounds below don't run into problems:
        ptr+cellsize>size(PJ,2) && error("cell plan does not agree with basic cellsizes")
        for m in 1:cellsize
            @simd for j in 1:size(Pis, 1)
                @inbounds Pis[j, i] += PJ[j,ptr+m]
            end
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
    truncate!(A, [μJ, ] threshold)

Drop values of `ν` smaller or equal than `threshold`. Acts inplace.
If μJ is given and A is a matrix, call
```julia 
truncate!(A[:,i], μJ[i]*threshold)
```
for each column of A.
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
""" 
    update_cell_plan!(P::DomDecPlan, PJ, I, k, j, μJ[;
            balance = true, truncate = true,
            truncate_Ythresh = 1e-16, truncate_Ythresh_rel=true])

Update cell `j` of partition `k` with the given cell plan `PJ`.
"""
function update_cell_plan!(P::DomDecPlan, PJ, I, k, j, μJ;
    balance = true, truncate = true,
    truncate_Ythresh = 1e-16, truncate_Ythresh_rel=true)
    
    # Compute target cell masses
    P_basic, mJ = reduce_cellplan_to_cellmarginals(P, PJ, k, j, μJ)

    if balance
        balance!(P_basic, mJ, truncate_Ythresh)
    end
    if truncate
        if truncate_Ythresh_rel
            truncate!(P_basic, mJ, truncate_Ythresh)
        else
            truncate!(P_basic, truncate_Ythresh)
        end
    end

    update_cell_plan!(P, P_basic, k, j, I)
    # TODO: efficient way, instead of creating sparsevec in each iteration?
    nothing
end

"""
    update_cell_plan!(P::DomDecPlan, P_basic, k, j, I)

Update cell `j` of partition `k` of the plan `P` using the columns
in `P_basic` as the basic cell marginals. `I` is the real support of
the columns of P_basic
"""
# function update_cell_plan!(P::DomDecPlan, P_basic, k, j, I)
#     basic_cells = P.composite_cells[k][j]
#     println(I)
#     for i in eachindex(basic_cells)
#         # Construct sparse vectors only with the positive entries
#         Ip = findall(>(0), @views P_basic[:,i])
#         P.gamma[basic_cells[i]] = sparsevec(I[Ip], P_basic[Ip,i], npoints(P.nu))
#     end
# end
# Second version, reusing basic cell marginals sparsevector
function update_cell_plan!(P::DomDecPlan, P_basic, k, j, I)
    basic_cells = P.composite_cells[k][j]
    for i in eachindex(basic_cells)
        νi = P.gamma[basic_cells[i]]
        empty!(νi.nzval)
        empty!(νi.nzind)
        for j in 1:size(P_basic,1)
            if P_basic[j,i] > 0
                push!(νi.nzind, I[j])
                push!(νi.nzval, P_basic[j,i])
            end
        end
    end
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

"""
    reduce_to_cellsize(γ0::AbstractMatrix, basic_cells)

Add together marginals corresponding to the same basic cell.
"""
function reduce_to_cells(K::AbstractMatrix, basic_cells)
    gamma = Vector{SparseVector{Float64, Int}}(undef, length(basic_cells))
    for i in eachindex(basic_cells)
        v = @views sum(K[:, basic_cells[i]], dims=2)
        gamma[i] = sparsevec(v)
    end
    return gamma
end

# TODO: Test this in iterate, solving with sinkhorn and with domdec
"""
    reduce_to_cells(gamma, gridshape, cellSize)

Add together marginals corresponding to the same basic cell, 
for given `shapeX` and `cellSize`.
"""
function reduce_to_cells(gamma, gridshape, cellsize)
    basic_cells, _ = get_cells(gridshape, cellsize)
    reduce_to_cellsize(gamma, basic_cells)
end

function DomDecPlan(mu::AbstractMeasure{D}, 
                        nu::AbstractMeasure, 
                        A::AbstractMatrix,
                        cellsize::Int, 
                        basic_cells::Vector, 
                        composite_cells::Vector, 
                        partitions::Vector,
                        alphas::Vector, 
                        betas::Vector, 
                        epsilon=1.0, 
                        partk=0; 
                        consistency_check = true) where D
    
    DomDecPlan(mu, nu, reduce_to_cells(A, basic_cells), 
                    cellsize, basic_cells, 
                    composite_cells, partitions,
                    epsilon, partk; consistency_check)
end
    
"""
    plan_to_dense_matrix(P, c[, balancing = true])

Turn P into a dense matrix using the dual potentials of the last iteration.
"""
function plan_to_dense_matrix(P, c, balancing = true)
    k = P.partk
    if k == 0; k = 1; end
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
"""
    plan_to_sparse_matrix(P, c[, k, balancing = true])

Turn P into a sparse matrix using the dual in the last iteration.
"""
function plan_to_sparse_matrix(P, c, balancing = true, truncate_Ythresh = 0)
    # This follows very closely MultiScaleOT.refine_support
    k = P.partk
    if k == 0; k = 1; end
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