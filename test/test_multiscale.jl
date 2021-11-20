# CODE STATUS: TESTS NOT DOCUMENTED
using SparseArrays
import MultiScaleOT as MOT 
import MultiScaleOT: AbstractMeasure, GridMeasure, npoints, normalize!
import DomDecOT as DD
import DomDecOT: DomDecPlan
import Random
import StatsBase: mean

@testset ExtendedTestSet "refine_submeasure" begin
    # 1D
    n = 4
    v = sparsevec([0, 1, 0, 0.3])
    u = ones(n)
    u2 = ones(2n)./2
    v2 = sparsevec([0, 0, 0.5, 0.5, 0, 0, 0.15, 0.15])
    refinements, _ = MOT.get_cells((2n,), 2)
    nzind, nzval = DD.refine_submeasure(v,u,u2,refinements)
    new_v2 = sparsevec(nzind, nzval, length(v2))
    @test new_v2 == sparsevec(v2)

    # 2D
    v = [1, 2, 1, 0.]
    u = [2, 2, 2, 2.]
    u2 = [0,0,0,2,0,2,0,2,2.]
    v2 = [0,0,0,1,0,2,0,1,0.]
    refinements, _ = MOT.get_cells((3,3), 2)
    nzind, nzval = DD.refine_submeasure(sparsevec(v),u,u2,refinements)
    new_v2 = sparsevec(nzind, nzval, length(v2))
    @test new_v2 == sparsevec(v2)
end

@testset ExtendedTestSet "refine_plan" begin
    # 1D
    gridshapes_1D = [
        [(16,), (16,)],
        [(13,), (17,)],
        [(43,), (32,)]
    ]

    gridshapes_2D = [
        [(16,16), (16,16)],
        [(100, 32), (23, 43)],
        [(27, 21), (65, 21)]
    ]

    # TODO: test 3D as well
    gridshapes_3D = []

    gridshapes = vcat(gridshapes_1D, gridshapes_2D, gridshapes_3D)

    cellsizes = [1,2,4]
    for (gridshapeX, gridshapeY) in gridshapes
        μ = rand(gridshapeX...)[:]; MOT.normalize!(μ)
        xs = [collect(1. :N) for N in gridshapeX]
        X = MOT.flat_grid(xs...)
        mu = MOT.GridMeasure(X, μ, gridshapeX)
        
        ν = rand(gridshapeY...)[:]; MOT.normalize!(ν)
        ys = [collect(1. :N) for N in gridshapeY]
        Y = MOT.flat_grid(ys...)
        nu = MOT.GridMeasure(Y, ν, gridshapeY)
        # Setup variables
        depth_mu = MOT.compute_multiscale_depth(mu)
        depth_nu = MOT.compute_multiscale_depth(nu)

        # Adjust to minimum depth
        depth = min(depth_mu, depth_nu)

        muH = MOT.MultiScaleMeasure(mu; depth)
        nuH = MOT.MultiScaleMeasure(nu; depth)

        # Generate DomDecPlan
        k = depth-1
        mu = muH[k]
        nu = nuH[k]
        A = nu.weights .* mu.weights'
        for cellsize in cellsizes
            P = DD.DomDecPlan(mu, nu, A, cellsize)

            new_P = DD.refine_plan(P, muH, nuH, k+1; consistency_check = true)
            # Check marginals are ok
            @test MOT.l1(new_P.mu.weights, muH[end].weights) < 1e-8
            @test MOT.l1(new_P.nu.weights, nuH[end].weights) < 1e-8
            @test new_P.mu.points == muH[end].points
            @test new_P.nu.points == nuH[end].points
            # Rest is taken care of by the consistency_check part
        end
    end
end







