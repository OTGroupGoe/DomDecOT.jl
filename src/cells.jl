# CODE STATUS: NOT REVISED, NOT TESTED

import MultiScaleOT: get_cells

"""
    get_basic_and_composite_cells(gridshape, cellsize)

Get the basic and composite cells corresponding to partitioning
a grid of shape `gridshape` into basic cells of size 
at most `cellsize × ... × cellsize`, and then joining
groups of 2^D adjacent basic cells together to form composite cells `compA`.
`compB` is obtained in an analogous manner, but with an offset of 
1 along each axes.

# Examples

""" 
function get_basic_and_composite_cells(gridshape, cellsize)
    basic_cells, cells_shape = get_cells(gridshape, cellsize)
    compA, _ = get_cells(cells_shape, 2)
    compB, _ = get_cells(cells_shape, 2, 1)
    composite_cells = [compA, compB]
    return basic_cells, composite_cells
end

"""
    get_partition(basic_cells, composite_cells)

Compute a partition by joining the basic cells corresponding
to each composite cells. 

# Examples
```julia-repl
julia> DD.get_partition([[1], [2, 3], [4]], [[1, 2], [3]])
2-element Vector{Vector{Int64}}:
 [1, 2, 3]
 [4]
```
"""
function get_partition(basic_cells, composite_cells)
    return [vcat(basic_cells[J]...) for J in composite_cells]
end

