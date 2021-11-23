# CODE STATUS: TESTS DOCUMENTED
import DomDecOT as DD
import DomDecOT: DomDecPlan
import MultiScaleOT as MOT
import MultiScaleOT: normalize!
import Random
using SparseArrays

# Initialize random marginals and product coupling
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

@testset ExtendedTestSet "fix_beta!" begin
    β0 = rand(10)

    # When the size is right
    β = copy(β0)
    was_alright = DD.fix_beta!(β, 10)
    @test (β == β0) & (was_alright = true)

    # When the size is smaller than needed
    β = copy(β0)
    was_alright = DD.fix_beta!(β, 20)
    @test (β == zeros(20)) & (was_alright == false)

    # When the size is bigger than needed
    β = copy(β0)
    was_alright = DD.fix_beta!(β, 5)
    @test (β == zeros(5)) & (was_alright == false)

    # TODO: test new method
    M = 32
    N = 33
    C = [MOT.l22(i,j) for j in 1.:N, i in 1:M]

    μJ = rand(M) .+ 1e-2; MOT.normalize!(μJ)
    νJ = rand(N) .+ 1e-2; MOT.normalize!(νJ)
    νI = rand(N) .+ 1e-2
    for ε in [1., 0.001]
        α = zeros(M)
        β = zeros(N)
        K = copy(C)

        DD.domdec_logsinkhorn!(β, α, νJ, νI, μJ, K, ε; max_iter = 1, verbose = false)
        β2 = Float64[]
        DD.fix_beta!(β2, α, C, νJ, νI, μJ, ε)
        @test MOT.l1(β, β2) < 1e-8 
    end
end

@testset ExtendedTestSet "scores" begin
    # Test that the scores functions yield the same values as
    # when computing the whole plan via a normal sinkhorn
    for N in [8, 13, 32]
        for cellsize in [1, 2, 4]
            # Initialize variables
            mu, nu, gamma = random_setup_domdec(N, cellsize)
            P0 = DomDecPlan(mu, nu, gamma, cellsize) 

            J = collect(1:N)
            I = collect(1:N)
            c = MOT.l22
            C = DD.get_cost_matrix(P0, c, J, I)
            ε = N/10
            P0.epsilon = ε

            μ = mu.weights
            ν = nu.weights
            X = mu.points
            Y = nu.points

            α = zeros(N)
            β = zeros(N)

            K = DD.get_kernel(C, β, α, ν, μ, ε)

            # Run the sinkhorn algorithm for enough iterations
            MOT.sinkhorn_stabilized!(β, α, ν, μ, K, ε; 
                                    max_error = 1e-10, 
                                    max_iter = 100000)  

            # Construct the plan from this already optimal plan
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
            P.partk = 1
            # Check that the scores give the same result 
            # for K and P
            p_score_1 = MOT.primal_score_dense(K, c, Y, X, ν, μ, ε)
            p_score_2 = DD.primal_score(P, c, ε)
            @test p_score_1 ≈ p_score_2

            d_score_1 = MOT.dual_score_dense(α, β, c, X, Y, μ, ν, ε)
            d_score_2 = DD.dual_score(P, c, ε)
            @test d_score_1 ≈ d_score_2

            @test DD.PD_gap(P, c, ε) < 1e-8
        end
    end
end