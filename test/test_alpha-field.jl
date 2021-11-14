# CODE STATUS: TESTS NOT DOCUMENTED
using SparseArrays
import MultiScaleOT as MOT 
import MultiScaleOT: AbstractMeasure, GridMeasure, npoints, normalize!
import DomDecOT as DD
import DomDecOT: DomDecPlan
import Random
import StatsBase: mean

function random_setup_domdec(N, cellsize)
    x1 = collect(1:N)
    X = MOT.flat_grid(x1)
    Y = MOT.flat_grid(x1)
    μ = rand(N) .+ 1e-2; normalize!(μ)
    ν = rand(N) .+ 1e-2; normalize!(ν)

    mu = MOT.GridMeasure(X, μ, (N,))
    nu = MOT.GridMeasure(Y, ν, (N,))

    gamma0 = ν .* μ'
    basic_cells, _ = DD.get_cells((N,), cellsize)
    gamma = [sparse(sum(gamma0[:,J], dims = 2)[:]) for J in basic_cells] # Only basic cell marginals remain
    return mu, nu, gamma
end

Random.seed!(0)

@testset ExtendedTestSet "smooth_alpha_field" begin
    # See that we are able to reconstruct a field where 
    # we have shifted each composite cell by a random offset
    # and there is a bit of noise.
    N = 8
    cellsize = 2
    P = DomDecPlan(random_setup_domdec(N, cellsize)..., cellsize)
    
    α = rand(N)
    δ = 1e-6 # error scale 
    for i in eachindex(P.partitions) 
        for j in eachindex(P.partitions[i])
            J = P.partitions[i][j]
            # A constant offset O(1)
            offset = rand()
            # Some noise of O(δ)
            noise = δ.*(rand(length(J)).-0.5)
            P.alphas[i][j] = α[J] .+ offset .+ noise
        end
    end

    α2 = DD.smooth_alpha_field(P)
    # Compare them up to a constant
    @test MOT.l1(α.- mean(α), α2 .- mean(α2)) < δ*N

    # TODO: test in 2D

end

@testset ExtendedTestSet "smooth_alpha_and_beta_fields" begin
    # Try a range of different parameters, including odd resolutions
    for N in [8, 13, 31]
        for cellsize in [1, 2, 3, 4]
            mu, nu, gamma = random_setup_domdec(N, cellsize)
            P0 = DomDecPlan(mu, nu, gamma, cellsize) 

            J = collect(1:N)
            I = collect(1:N)
            c = MOT.l22
            C = DD.get_cell_cost_matrix(P0, c, J, I)
            ε = N/10
            P0.epsilon = ε

            u = ones(N)
            v = ones(N)
            μ = mu.weights
            ν = nu.weights
            K = DD.get_kernel(C, 0, 0, ν, μ, ε)
            KT = K'

            # Run the sinkhorn algorithm for enough iterations
            MOT.sinkhorn!(v, u, ν, μ, K, KT, 10000)

            K = v .* K .* u' # Scale it

            α = ε.*log.(u)
            β = ε.*log.(v)

            P = DD.DomDecPlan(mu, nu, K, cellsize, ε)
            for i in eachindex(P.partitions) 
                for j in eachindex(P.partitions[i])
                    J = P.partitions[i][j]
                    offset = rand()
                    P.alphas[i][j] = α[J] .+ offset
                    _, I = DD.get_cell_Y_marginal(P, i, j)
                    P.betas[i][j] = β[I] .- offset
                end
            end

            α2, β2 = DD.smooth_alpha_and_beta_fields(P, c)
            K2 = DD.get_kernel(C, β2, α2, ν, μ, ε)
            @test MOT.l1(K, K2) < 1e-8
        end
    end
end