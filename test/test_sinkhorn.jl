# CODE STATUS: TESTS DOCUMENTED
using SparseArrays
import MultiScaleOT as MOT 
import MultiScaleOT: normalize!
import DomDecOT as DD
import Random
import LinearAlgebra: dot

Random.seed!(0)

@testset ExtendedTestSet "domdec_sinkhorn" begin
    # We solve entropic OT with the sinkhorn from the MOT library
    # and with the domdec_sinkhorn, that does some preprocessing and 
    # then calls it. 
    # 
    # The Kernels can be different, so it is reassuring to check
    # that the result must be the same 
    N = 8
    cellsize = 1
    x1 = collect(1:N)
    X = MOT.flat_grid(x1)
    Y = MOT.flat_grid(x1)
    μ = rand(N) .+ 1e-2; normalize!(μ)
    ν = rand(N) .+ 1e-2; normalize!(ν)
    # Dummy cell Y-marginal
    νJ = rand(N)
    normalize!(μ, sum(νJ))

    c = MOT.l22
    C = [c(x,y) for y in eachcol(X), x in eachcol(Y)]
    ε = N/10

    # Solve first time scaling the kernel with νJ .* μ'
    K1 = DD.get_kernel(C, 0, 0, νJ, μ, ε)

    a1 = zeros(N)
    b1 = zeros(N)

    MOT.sinkhorn_stabilized!(b1, a1, νJ, μ, K1, ε)  
    
    # Solve second time with dedicated domdec solver, 
    # which scales the kernel with ν .* μ'
    K2 = copy(C)

    a2 = zeros(N)
    b2 = zeros(N)
    DD.domdec_sinkhorn_stabilized!(b2, a2, νJ, ν, μ, K2, ε)

    # Transport cost must agree
    @test dot(K1, C) ≈ dot(K2, C)
    # <μ, α> must agree
    @test dot(μ, a1) ≈ dot(μ, a2)
    # <νJ, β> must agree, where each β must be adjusted by the 
    # difference in their kernel scaling
    @test dot(νJ, b1.+ ε.*log.(νJ)) ≈ dot(νJ, b2 .+ ε.*log.(ν))
end
