using DomDecOT
import DomDecOT as DD

@testset ExtendedTestSet "get partition" begin
    # 1D 
    basic_cells = [[1], [2], [3], [4]]
    comp_cells = [[1], [2, 3], [4]]
    @test DD.get_partition(basic_cells, comp_cells) == [[1], [2, 3], [4]]

    basic_cells, cells_shape = DD.get_cells((8, ), 2, 0)
    comp_cells, _ = DD.get_cells(cells_shape, 1, 0)
    @test DD.get_partition(basic_cells, comp_cells) == [[1,2],[3,4],[5,6],[7,8]]

    comp_cells, _ = DD.get_cells(cells_shape, 2, 0)
    @test DD.get_partition(basic_cells, comp_cells) == [[1,2,3,4],[5,6,7,8]]

    # 2D 
    basic_cells, cells_shape = DD.get_cells((8, 8), 2, 0)
    comp_cells, _ = DD.get_cells(cells_shape, 2, 0)
    partition = DD.get_partition(basic_cells, comp_cells)
    # By design the parition is not sorted, otherwise we couldn't transmit
    # information from composite to basic cells.  
    @test sort.(partition) == DD.get_cells((8, 8), 4, 0)[1]
    
    comp_cells, _ = DD.get_cells(cells_shape, 2, 1)
    partition = DD.get_partition(basic_cells, comp_cells)
    # By design the parition is not sorted, otherwise we couldn't transmit
    # information from composite to basic cells.  
    @test sort.(partition) == DD.get_cells((8, 8), 4, 2)[1]

    # 3D: TODO
end