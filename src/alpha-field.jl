# CODE STATUS: MOSTLY TESTED, REVISED
import LinearAlgebra: dot
# TODO: get_discrete_gradient in MultiScaleOT or here?
import MultiScaleOT: get_discrete_gradient
"""
    get_alpha_field(P, k)

Glue cell duals of partition `k` to form a global dual.
"""
function get_alpha_field(P::DomDecPlan{GridMeasure{D},M}, k) where {D,M}
    A = zeros(P.mu.gridshape) 
    for (i,J) in enumerate(P.partitions[k])
        A[J] = P.alphas[k][i]
    end
    return A # A has shape `gridshape`
end

# This version should be more efficient but
# actually allocates too much
# function get_alpha_diff(P::DomDecPlan{D}) where D
#     shapeX = P.shapeX
#     alpha_diff = zeros(P.shapeX)
#     for (i,J) in enumerate(P.partitions[1])
#         @views alpha_diff[J] .+= P.alphas[1][i]
#     end
    
#     for (i,J) in enumerate(P.partitions[2])
#         @views alpha_diff[J] .-= P.alphas[2][i]
#     end
#     return alpha_diff
# end

"""
    get_alpha_graph(P::DomDecPlan{2}, alpha_diff, cellsize)
   
Compute Helmholtz decomposition of `alpha_diff`, averaging first on
basic cells. 
"""
function get_alpha_graph(P::DomDecPlan{GridMeasure{1}, M}, alpha_diff) where M
    basic_cell_length = Int(ceil(P.mu.gridshape[1] / P.cellsize))
    
    basic_diff = zeros(basic_cell_length)
    
    # Average difference in composite cell
    for (i, B) in enumerate(P.basic_cells)
        # @views basic_diff[i] = mean(alpha_diff[B]) # Remove if the following works well
        μB = view_X_marginal(P, B)
        basic_diff[i] = dot(alpha_diff[B], μB) / sum(μB)
    end
    
    pad_basic_length = basic_cell_length + Int(isodd(basic_cell_length))
    
    # If the number of basic cells along each dimension is not even, 
    # we pad `basic_cell_shapes` with a copy of the last row/column. 
    # That way the graph weights can be computed fairly easily.
    if isodd(basic_cell_length)
        push!(basic_diff, basic_diff[end])
    end

    # TODO: Probably this weight-computing to dedicated function
    # For D=1
    Nx = length(basic_diff)
    NWx = Nx÷2
    lastx = Nx-1

    # TODO: Weight with distance between points or leave as is?
    b = @views basic_diff[3:2:Nx-1] .- basic_diff[2:2:Nx-2] 
    
    # Build system and solve
    AT = get_discrete_gradient(NWx)
    # remove first colum of A (row of AT) to make A injective
    AT = AT[2:end, :]

    # Solve system (A'*A)*V
    A = AT'
    ATA = AT * A
    ATb = AT * b

    # Solve
    V = ATA \ ATb
    # Set V[0] to zero
    pushfirst!(V, 0)
    return V
end

# TODO: generalize to arbitrary dimension
"""
    get_alpha_graph(P::DomDecPlan{2}, alpha_diff)
   
Compute Helmholtz decomposition of `alpha_diff`, averaging first on
basic cells. 
"""
function get_alpha_graph(P::DomDecPlan{GridMeasure{2}, M}, alpha_diff) where M
    basic_cell_shapes = Int.(ceil.(P.mu.gridshape ./ P.cellsize))
        
    basic_diff = zeros(basic_cell_shapes)
    
    # Average difference in composite cell
    for (i, B) in enumerate(P.basic_cells)
        # @views basic_diff[i] = mean(alpha_diff[B]) # Remove if the following works well
        μB = view_X_marginal(P, B)
        basic_diff[i] = dot(alpha_diff[B], μB) / sum(μB)
    end
    
    pad_basic_shape = basic_cell_shapes .+ isodd.(basic_cell_shapes)
    
    # If the number of basic cells along each dimension is not even, 
    # we pad `basic_cell_shapes` with a copy of the last row/column. 
    # That way the graph weights can be computed fairly easily.
    if pad_basic_shape != basic_cell_shapes
        Mx, My = basic_cell_shapes
        Nx, Ny = pad_basic_shape
        pad_basic_diff = zeros(pad_basic_shape)
        pad_basic_diff[1:Mx, 1:My] .= basic_diff
        if Mx < Nx
            @views pad_basic_diff[end, 1:end] .= pad_basic_diff[end-1, 1:end]
        end
        if My < Ny
            @views pad_basic_diff[1:end, end] .= pad_basic_diff[1:end, end-1]
        end
        basic_diff = pad_basic_diff
    end

    # TODO: This seems to be possible to do pretty efficiently
    # with some kind of convolutional pass
    # TODO: Probably this weight-computing to dedicated function
    # For D=2
    Nx, Ny = size(basic_diff)
    NWx, NWy = Nx÷2, Ny÷2 
    lastx, lasty  = Nx-1, Ny-1

    Wx = reshape(basic_diff[2:lastx,:], 2, NWx-1, 2, NWy)
    Wx[1,:,:,:] .*= -1
    WWx = reshape(sum(Wx, dims = (1, 3))./2, NWx-1, NWy) # sum in one dimension, mean in another is the same as suming and dividing

    Wy = reshape(basic_diff[:, 2:lasty], 2, NWx, 2, NWy-1)
    Wy[:,:,1,:] .*= -1
    WWy = reshape(sum(Wy, dims = (1, 3))./2, NWx, NWy-1)
    WWx

    # Build system and solve
    gradx, grady = get_discrete_gradient(NWx, NWy)
    # Combine
    AT = [gradx grady]
    # remove first colum of A (row of AT) to make A injective
    AT = AT[2:end, :]

    # Solve system (A'*A)*V
    A = AT'
    ATA = AT * A
    b = [WWx[:]; 
        WWy[:]]
    ATb = AT * b

    # Solve
    V = ATA \ ATb
    # Set V[0] to zero
    pushfirst!(V, 0)
    return V
end

"""
    smooth_alpha_field(P::DomDecPlan, cellsize)

Compute a smooth version of the `X` dual of `P` by performing a 
Helmholtz decomposition (see https://arxiv.org/abs/2001.10986 for details)
"""
function smooth_alpha_field(P::DomDecPlan{GridMeasure{D}, M}) where {D,M}
    # TODO: this right now allocates too much, go for simple subtraction
    #alpha_diff = get_alpha_diff(P)
    alpha_field = get_alpha_field(P, 1)
    alpha_diff = alpha_field .- get_alpha_field(P, 2)
    # Get Helmholt decomposition
    V = get_alpha_graph(P, alpha_diff)

    for (i, J) in enumerate(P.partitions[1])
        alpha_field[J] .-= V[i]
    end
    return alpha_field
end

"""
    smooth_alpha_and_beta_field(P::DomDecPlan, cellsize)

Compute a smooth version of the duals of `P` by performing a 
Helmholtz decomposition on `X` and adapting the dual in `Y`
accordingly.
"""
function smooth_alpha_and_beta_fields(P::DomDecPlan{GridMeasure{D}, M}, c) where {D,M}
    alpha_field = get_alpha_field(P, 1)
    alpha_diff = alpha_field .- get_alpha_field(P, 2)
    # Get Helmholtz decomposition
    V = get_alpha_graph(P, alpha_diff)

    for (i, J) in enumerate(P.partitions[1])
        alpha_field[J] .-= V[i]
    end

    # Compute beta
    beta_field = zeros(npoints(P.nu))
    # V is computed with respect to the first partition
    k0 = 1
    for (i,J) in enumerate(P.partitions[k0])
        νJ, I = get_cell_Y_marginal(P, k0, i)
        νI = get_Y_marginal(P, I)
        # Weight each entry by its amount of mass, so that conflicting betas
        # can vote
        βJ = get_cell_beta(P, k0, i)
        # If βJ is broken, we need more data to fix it
        if length(βJ) != length(I)
            αJ = get_cell_alpha(P, k0, i)
            C = get_cell_cost_matrix(P, c, J, I)
            μJ = view_X_marginal(P, J)
            ε = P.epsilon
            fix_beta!(βJ, αJ, C, νJ, νI, μJ, ε)
        end

        beta_field[I] .+= (V[i] .+ βJ).* (νJ./νI) 
    end

    return alpha_field, beta_field
end
