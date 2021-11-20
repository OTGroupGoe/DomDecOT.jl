# TODO: get_discrete_gradient in MultiScaleOT or here?

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

"""
    fix_beta!(β, α, C, νJ, νI, μJ, ε)

Set β to the conjugate of α, taking into account the cell parameters.
Stable implementation.
"""
function fix_beta!(β, α, C, νJ, νI, μJ, ε)
    M, _ = size(C)
    fix_beta!(β, M)
    α .+= ε .* log.(μJ)
    β .= ε .* log.(νJ./νI) .+ MultiScaleOT.logsumexp(C, α, ε, 1)
    α .-= ε .* log.(μJ)
    nothing
end

########################################
# Obtaining global dual variables
########################################

# Discrete gradients 

# TODO: Test
# TODO, HIGH, CORRECTNESS
# Pass `x` instead of `nx` to the `get_discrete_gradient` functions.
# and then return a matrix that works for unevenly spaced grids.
"""
    get_discrete_gradient(nx)

Get (transpose of) X-gradient matrices for a one dimensional lattice of size `nx`.
"""
function get_discrete_gradient(nx)
    D = ones(nx-1)
    return spdiagm(nx, nx-1, -1=> D, 0=>-D)
end

"""
    get_discrete_gradient(nx, ny)

Get (transpose of) X- and Y-gradient matrices for a discrete graph of size `(nx, ny)`.

# Relation with the Python library
If we define in Julia
```julia
gx, gy = get_discrete_gradient(nx, ny)
```
and in Python
```python
GX, GY = Common.getDiscreteGradients(ny, nx)
```
Then `GX, GY == gx', gy'`
"""
function get_discrete_gradient(nx, ny)
    index_array = reshape(collect(1:nx*ny), nx, ny)
    
    # compute gradx
    colptr = collect(1:2:2*(nx-1)*ny+1)
    data = ones((nx-1)*ny*2)
    data[1:2:end] .= -1
    rowval = zeros(Int, (nx-1)*ny*2)
    @views rowval[2:2:end] .= index_array[2:end, :][:]
    @views rowval[1:2:end] .= index_array[1:end-1, :][:]
    gradx = SparseMatrixCSC(nx*ny, (nx-1)*ny, colptr, rowval, data)

    # compute grady
    colptr = collect(1:2:2*nx*(ny-1)+1)
    data = ones(nx*(ny-1)*2)
    data[1:2:end] .= -1
    rowval = zeros(Int, nx*(ny-1)*2)
    @views rowval[2:2:end] .= index_array[:, 2:end][:]
    @views rowval[1:2:end] .= index_array[:, 1:end-1][:]
    grady = SparseMatrixCSC(nx*ny, nx*(ny-1), colptr, rowval, data)
    
    return gradx, grady
end

"""
    get_discrete_gradient(nx, ny, nz)

Get (transpose of) X-, Y- and Z-gradient matrices for a discrete graph of size `(nx, ny, nz)`.
"""
function get_discrete_gradient(nx, ny, nz)
    index_array = reshape(collect(1:nx*ny*nz), nx, ny, nz)
    # gradx
    colptr = collect(1:2:2*(nx-1)*ny*nz+1)
    data = ones((nx-1)*ny*nz*2)
    data[1:2:end] .= -1
    rowval = zeros(Int, (nx-1)*ny*nz*2)
    @views rowval[1:2:end-1] .= index_array[1:end-1, :, :][:]
    @views rowval[2:2:end] .= index_array[2:end, :, :][:]
    gradx = SparseMatrixCSC(nx*ny*nz, (nx-1)*ny*nz, colptr, rowval, data)

    # grady
    colptr = collect(1:2:2*nx*(ny-1)*nz+1)
    data = ones(nx*(ny-1)*nz*2)
    data[1:2:end] .= -1
    rowval = zeros(Int, nx*(ny-1)*nz*2)
    @views rowval[1:2:end-1] .= index_array[:, 1:end-1, :][:]
    @views rowval[2:2:end] .= index_array[:, 2:end, :][:]
    grady = SparseMatrixCSC(nx*ny*nz, nx*(ny-1)*nz, colptr, rowval, data)
    
    # gradz
    colptr = collect(1:2:2*nx*ny*(nz-1)+1)
    data = ones(nx*ny*(nz-1)*2)
    data[1:2:end] .= -1
    rowval = zeros(Int, nx*ny*(nz-1)*2)
    @views rowval[1:2:end-1] .= index_array[:, :, 1:end-1][:]
    @views rowval[2:2:end] .= index_array[:, :, 2:end][:]
    gradz = SparseMatrixCSC(nx*ny*nz, nx*ny*(nz-1), colptr, rowval, data)
    return gradx, grady, gradz
end

# TODO, LOW, COMPLETENESS
# General version of `get_discrete_gradient` (for multidimensional data)?

# Global duals

# TODO, MEDIUM, CONSISTENCY: 
# Probably the global duals should be one-dimensional arrays,
# and only be converted to D-dimensional when required by the 
# refinement algorithm.

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

"""
    get_alpha_graph(P::DomDecPlan, alpha_diff, cellsize)
   
Compute Helmholtz decomposition on, `alpha_diff`, averaging on
basic cells, as explained in https://arxiv.org/abs/2001.10986, Section 6.3.
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

Compute a smooth version of the `X` dual of `P` by performing a Helmholtz decomposition 
(see https://arxiv.org/abs/2001.10986, Section 6.3 for details)
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
    smooth_alpha_and_beta_field(P::DomDecPlan, c)

Compute a smooth version of the duals of `P` by performing a Helmholtz decomposition
(see https://arxiv.org/abs/2001.10986, Section 6.3 for details)
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
        αJ = get_cell_alpha(P, k0, i)
        C = get_cost_matrix(P, c, J, I)
        μJ = view_X_marginal(P, J)
        ε = P.epsilon
        fix_beta!(βJ, αJ, C, νJ, νI, μJ, ε)

        beta_field[I] .+= (V[i] .+ βJ).* (νJ./νI) 
    end

    return alpha_field, beta_field
end
