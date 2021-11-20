import DomDecOT as DD
import MultiScaleOT as MOT

@testset ExtendedTestSet "display" begin
    io = IOBuffer()
    # GridMeasure
    mu = MOT.GridMeasure([1. 1], [1., 1], (2,))
    nu = MOT.CloudMeasure([1 2; 1 1.], [1,1.])
    P = DD.DomDecPlan(mu, nu, [1 0; 0 1.], 1)
    
    show(io, P)
    @test String(take!(io)) == "DomDecPlan with cellsize 1 and marginals\n"*
                               "    mu: 1D GridMeasure with gridshape (2,)\n"*
                               "    nu: 2D CloudMeasure with 2 stored entries in the box\n   [1.0, 2.0] × [1.0, 1.0]"
end



