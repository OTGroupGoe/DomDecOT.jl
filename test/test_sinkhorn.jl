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
    K = copy(C)

    a = zeros(N)
    b = zeros(N)
    kwargs = (; max_error = 1e-10, max_iter = 10000)
    DD.domdec_sinkhorn_stabilized!(b, a, νJ, ν, μ, K, ε; kwargs...)

    # <νJ, β> must agree, where each β must be adjusted by the 
    # difference in their kernel scaling
    b .+= ε.*log.(ν./νJ) 
    gap = MOT.PD_gap_dense(b, a, K, c, Y, X, νJ, μ, ε)
    @test abs(gap) <1e-8
end

@testset ExtendedTestSet "domdec-sinkhorn-autofix" begin
    # This should converge even for very small values of ε
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
    K = copy(C)

    a = zeros(N)
    b = zeros(N)
    DD.domdec_sinkhorn_autofix!(b, a, νJ, ν, μ, K, ε; verbose = true)

    # scaling used in MOT is different, adjust
    b .+= ε.*log.(ν./νJ) 
    @test abs(MOT.PD_gap_dense(b, a, K, c, Y, X, νJ, μ, ε)<1e-6)
end
