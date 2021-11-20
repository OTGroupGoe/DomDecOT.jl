using Base: product
using SparseArrays
import DomDecOT as DD
import DomDecOT: DomDecPlan
using LinearAlgebra: norm
import MultiScaleOT as MOT
using MultiScaleOT
import Random

function product_setup_domdec(N, cellsize)
    x1 = collect(1:N)
    X = MOT.flat_grid(x1)
    Y = MOT.flat_grid(x1)
    μ = ones(N); normalize!(μ)
    ν = ones(N); normalize!(ν)

    mu = MOT.GridMeasure(X, μ, (N,))
    nu = MOT.GridMeasure(Y, ν, (N,))

    gamma0 = ν .* μ'
    basic_cells, _ = DD.get_cells((N,), cellsize)
    gamma = [sparse(sum(gamma0[:,J], dims = 2)[:]) for J in basic_cells] # Only basic cell marginals remain
    return mu, nu, gamma
end

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

function identity_setup_domdec(N, cellsize)
    x1 = collect(1:N)
    X = MOT.flat_grid(x1)
    Y = MOT.flat_grid(x1)
    μ = ones(N); normalize!(μ)
    ν = ones(N); normalize!(ν)

    mu = MOT.GridMeasure(X, μ, (N,))
    nu = MOT.GridMeasure(Y, ν, (N,))

    gamma0 = [Int(xi == yj) for yj in eachcol(Y), xi in eachcol(X)]
    basic_cells, _ = DD.get_cells((N,), cellsize)
    gamma = [sparse(sum(gamma0[:,J], dims = 2)[:]) for J in basic_cells] # Only basic cell marginals remain
    return mu, nu, gamma
end

@testset ExtendedTestSet "test-defined functions" begin
    # Product setup
    Random.seed!(0)
    N = 4
    cellsize = 1
    mu, nu, gamma = product_setup_domdec(N, cellsize)
    μ = mu.weights
    ν = nu.weights
    @test Array(hcat(gamma...)) == ν .* μ'

    mu, nu, gamma = product_setup_domdec(N, 2)
    @test Array(hcat(gamma...)) == ν .* [μ[1]+μ[2] μ[3]+μ[4]]

    # Product random setup
    N = 4
    cellsize = 1
    mu, nu, gamma = random_setup_domdec(N, cellsize)
    μ = mu.weights
    ν = nu.weights
    @test Array(hcat(gamma...)) == ν .* μ'

    mu, nu, gamma = random_setup_domdec(N, 2)
    μ = mu.weights
    ν = nu.weights
    @test all(Array(hcat(gamma...)) .≈ ν .* [μ[1]+μ[2] μ[3]+μ[4]])

    # Diagonal setup 
    N = 4
    cellsize = 1
    mu, nu, gamma = identity_setup_domdec(N, cellsize)
    μ = mu.weights
    ν = nu.weights
    @test Array(hcat(gamma...)) == [
                                    1 0 0 0
                                    0 1 0 0
                                    0 0 1 0
                                    0 0 0 1
    ] 

    mu, nu, gamma = identity_setup_domdec(N, 2)
    μ = mu.weights
    ν = nu.weights
    @test Array(hcat(gamma...)) == [
                                    1 0
                                    1 0
                                    0 1
                                    0 1
    ] 

end

@testset ExtendedTestSet "DomDecPlan" begin
    N = 4
    cellsize = 1
    mu, nu, gamma = product_setup_domdec(N, cellsize)
    P = DomDecPlan(mu, nu, gamma, cellsize) 
    @test isa(P, DomDecPlan{MOT.GridMeasure{1}})

    # Test that no modification is done right now
    # TODO: Maybe drop zero entries in constructor?
    @test P.gamma == gamma
    @test P.mu == mu
    @test P.nu == nu

    # Check errors
    mu2 = MOT.GridMeasure(mu.points[:,2:end], mu.weights[2:end], mu.gridshape .-1)
    nu2 = MOT.GridMeasure(nu.points[:,2:end], nu.weights[2:end], nu.gridshape .-1)
    #@test_throws ErrorException("basic cells not compatible with number of atoms in mu") DomDecPlan(mu2, nu, gamma, cellsize)
    @test_throws ErrorException("size of second marginal of gamma doesn't match nu") DomDecPlan(mu, nu2, gamma, cellsize)

    # Marginal check
    gamma0 = deepcopy(gamma)
    gamma0[1][1] -= 0.01
    gamma0[2][1] += 0.01
    @test_throws ErrorException("first marginal of gamma does not match mu") DomDecPlan(mu, nu, gamma0, cellsize)

    gamma0 = deepcopy(gamma)
    gamma0[1][1] -= 0.01
    gamma0[1][2] += 0.01
    @test_throws ErrorException("second marginal of gamma does not match nu") DomDecPlan(mu, nu, gamma0, cellsize)


    # TODO: test rest of consistencies: passing bad cellsize, size of alhpas...
    # TODO: check also for cellSize > 1
end

@testset ExtendedTestSet "get_cell_X_massive_points" begin
    P = DomDecPlan(product_setup_domdec(4, 1)..., 1) 
    J = [1, 2, 3, 4]
    @test DD.get_X_massive_points(P, J) == J

    P.mu.weights[1] = 0
    @test DD.get_X_massive_points(P, J) == [2, 3, 4]
end

# TODO: build appropriate tests
# @testset ExtendedTestSet "get_cell_plan" begin
#     P = DomDecPlan(product_setup_domdec(4, 1)..., 1) 
#     J = [1, 2, 3, 4]
#     # TODO: these need to fail... or maybe not because basic cells are just of size 1?
#     @test DD.get_cell_plan(P, J) == P.γ

#     J2 = [1, 2]
#     @test DD.get_cell_plan(P, J2) == P.γ[J2]
# end

@testset ExtendedTestSet "get_cell_X_marginal" begin
    P = DomDecPlan(product_setup_domdec(4, 1)..., 1) 
    J = [1, 2, 3, 4]
    μJ = DD.get_X_marginal(P, J)
    @test  μJ == P.mu.weights

    J2 = [1, 2]
    μJ = DD.get_X_marginal(P, J2)
    @test μJ == P.mu.weights[J2]

    # Changing μJ does not change P:
    prev_value = μJ[1]
    μJ[1] = 0
    @test P.mu.weights[J2][1] == prev_value

    # TODO: test also cell versions
end

@testset ExtendedTestSet "view_cell_X_marginal" begin
    P = DomDecPlan(product_setup_domdec(4, 1)..., 1) 
    J = [1, 2, 3, 4]
    μJ = DD.view_X_marginal(P, J)
    @test all(μJ .== P.mu.weights)

    J2 = [1, 2]
    μJ = DD.view_X_marginal(P, J2)
    @test all(μJ .== P.mu.weights[J2])

    # Changing μJ does change P:
    prev_value = μJ[1]
    μJ[1] = 0
    @test P.mu.weights[J2][1] == 0

end

@testset ExtendedTestSet "get_Y_marginal, get_cell_Y_marginal" begin
    P = DomDecPlan(product_setup_domdec(4, 1)..., 1) 
    k = 1
    j = 1
    ν = P.nu.weights
    # The first cell of the first partition is [1,2], and since the initialization is the product one, 
    # we obtain just half of the total marginal
    νJ, I = DD.get_cell_Y_marginal(P, k, j)
    @test (I, νJ) == (collect(eachindex(ν)), ν/2)

    # Get the Y marginal supported on I
    νI = DD.get_Y_marginal(P, I)
    @test  νI == ν

    # View the Y marginal supported on I
    νI = DD.view_Y_marginal(P, I)
    @test  νI == ν
    # TODO: more tests?
end

@testset ExtendedTestSet "view_cell X and Y" begin
    P = DomDecPlan(product_setup_domdec(4, 1)..., 1) 
    J = [1, 2, 3, 4]
    I = [1, 2, 3, 4]
    @test DD.view_X(P, J) == P.mu.points
    @test DD.view_Y(P, I) == P.nu.points

    J2 = [1, 2]
    I2 = [3, 4]
    @test DD.view_X(P, J2) == P.mu.points[:, 1:2]
    @test DD.view_Y(P, I2) == P.nu.points[:, 3:4]
end

@testset ExtendedTestSet "get_cell_duals" begin
    mu, nu, gamma = product_setup_domdec(4, 1)
    cellsize = 1
    basic_cells, composite_cells = DD.get_basic_and_composite_cells(mu.gridshape, cellsize)
    partitions = [DD.get_partition(basic_cells, comp) for comp in composite_cells]
    alphas = [[rand(length(J)) for J in part] for part in partitions]
    betas = [[rand(length(J)) for J in part] for part in partitions]
    P = DomDecPlan(mu, nu, gamma, 
                    cellsize, basic_cells, 
                    composite_cells, partitions,
                    alphas, betas)
    
    @test DD.get_cell_alpha(P, 2, 1) == alphas[2][1]
    @test DD.get_cell_beta(P, 2, 1) == betas[2][1]
end

@testset ExtendedTestSet "get_cost_matrix" begin
    c = MOT.l22
    P = DomDecPlan(product_setup_domdec(4, 1)..., 1)    
    # Whole cost matrix would be
    # C = [
    #     0 1 4 9
    #     1 0 1 4
    #     4 1 0 1
    #     9 4 1 0
    # ] 
    # The cost matrix corresponding to first cell of first partition is
    C1 = [
        0 1
        1 0
        4 1
        9 4
    ]
    # The cost matrix corresponding to second cell of second partition is
    C2 = [
        1 4
        0 1
        1 0
        4 1
    ]
    I = [1, 2, 3, 4]
    @test DD.get_cell_cost_matrix(P, c, 1, 1, I) == C1

    @test DD.get_cell_cost_matrix(P, c, 2, 2, I) == C2
end

@testset ExtendedTestSet "get_cell_plan" begin
    # This only works after having performed some iterations
    N = 8
    cellsize = 1
    P = DomDecPlan(random_setup_domdec(N, cellsize)..., cellsize) 

    J = collect(1:N)
    I = collect(1:N)
    c = MOT.l22
    C = DD.get_cost_matrix(P, c, J, I)
    ε = P.epsilon

    u = ones(N)
    v = ones(N)
    μ = P.mu.weights
    ν = P.nu.weights
    K = DD.get_kernel(C, 0, 0, ν, μ, ε)
    KT = K'

    # Run the sinkhorn algorithm for enough iterations
    MOT.sinkhorn!(v, u, ν, μ, K, KT, 10000)

    K = v .* K .* u' # Scale it

    # Check PD gap
    α = ε.*log.(u)
    β = ε.*log.(v)
    @test abs(PD_gap_dense(β, α, K, c, I', J', ν, μ, ε) < 1e-8)
    
    # Set columns in P to those of k
    for i in eachindex(P.gamma)
        P.gamma[i] = sparsevec(K[:,i])
    end
    
    for i in eachindex(P.partitions) 
        for j in eachindex(P.partitions[i])
            J = P.partitions[i][j]
            offset = 0# rand()
            P.alphas[i][j] = α[J] .+ offset
            P.betas[i][j] = β .- offset
            K2, I = DD.get_cell_plan(P, c, i, j, false, 0)
            @test MOT.l1(K2, K[I,J]) < 1e-8
        end
    end

    # Get global plan
    K2 = DD.plan_to_dense_matrix(P, c, 1, false)
    @test MOT.l1(K2, K) < 1e-8
    K2 = DD.plan_to_dense_matrix(P, c, 2, false)
    @test MOT.l1(K2, K) < 1e-8
    # TODO: still should test these functions after real domdec iterations
end

@testset ExtendedTestSet "reduce_cellplan_to_cellmarginals" begin
    N = 8
    cellsize = 2
    P = DomDecPlan(random_setup_domdec(N, cellsize)..., cellsize) 
    # 1st elements of A-partition has 4 items 
    μJ = DD.get_cell_X_marginal(P, 1, 1)
    PJ = rand(N, 4)
    PJ_basic = [PJ[:,1].+PJ[:,2] PJ[:,3].+PJ[:,4]]
    mJ = [μJ[1]+μJ[2], μJ[3]+μJ[4]]
    @test (PJ_basic, mJ) == DD.reduce_cellplan_to_cellmarginals(P, PJ, 1, 1, μJ)
end

@testset ExtendedTestSet "truncate" begin
    P0 = DomDecPlan(product_setup_domdec(4, 1)..., 1) 

    # Truncate plan TODO: maybe remove?
    P = deepcopy(P0)
    DD.truncate!(P, 1e-8)
    @test all(P.gamma[i] == P0.gamma[i] for i in eachindex(P.gamma))

    # TODO: more testing?
end

@testset ExtendedTestSet "update_cell_plan" begin
    # First try the version with P_basic already reduced
    N = 8
    cellsize = 2
    P0 = DomDecPlan(random_setup_domdec(N, cellsize)..., cellsize) 
    P = deepcopy(P0)
    # Choose some cells
    for (k,j) in [(1,1), (1,2), (2,1), (2,2)]
        # Assume that we get only half of the entries >0
        P_basic = rand(N÷2, length(P.composite_cells[k][j]))
        I = [1,3,5,7]# Support of basic marginals on that subdomain
        DD.update_cell_plan!(P, P_basic, k, j, I)
        for (i, l) in enumerate(P.composite_cells[k][j])
            @test P.gamma[l].nzind == I
            @test P.gamma[l].nzval == P_basic[:,i]
        end
    end

    # And now test the version with P_basic not yet reduced
    P = deepcopy(P0)
    P = DomDecPlan(random_setup_domdec(N, cellsize)..., cellsize) 

    J = collect(1:N)
    I = collect(1:N)
    c = MOT.l22
    C = DD.get_cost_matrix(P, c, J, I)
    ε = P.epsilon

    u = ones(N)
    v = ones(N)
    μ = P.mu.weights
    ν = P.nu.weights
    K = DD.get_kernel(C, 0, 0, ν, μ, ε)
    KT = K'

    # Run the sinkhorn algorithm for enough iterations
    MOT.sinkhorn!(v, u, ν, μ, K, KT, 10000)

    K = v .* K .* u' # Scale it

    α = ε.*log.(u)
    β = ε.*log.(v)

    for i in eachindex(P.partitions) 
        for j in eachindex(P.partitions[i])
            J = P.partitions[i][j]
            P.alphas[i][j] = α[J]
            P.betas[i][j] = copy(β)
            μJ = DD.get_cell_X_marginal(P, i, j)
            PJ = K[:,J]
            
            DD.update_cell_plan!(P, PJ, I, i, j, μJ;
                        balance = false, truncate = false)
            
            PJ_again, _ = DD.get_cell_plan(P, c, i, j, false, 0)
            @test MOT.l1(PJ, PJ_again) < 1e-8 
        end
    end

end

@testset ExtendedTestSet "matrix-to-domdecplan" begin
    # This only works after having performed some iterations
    N = 8
    cellsize = 2
    mu, nu, gamma = random_setup_domdec(N, cellsize)
    P0 = DomDecPlan(mu, nu, gamma, cellsize) 

    J = collect(1:N)
    I = collect(1:N)
    c = MOT.l22
    C = DD.get_cost_matrix(P0, c, J, I)
    ε = P0.epsilon

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
            P.alphas[i][j] = α[J]
            P.betas[i][j] = β
        end
    end
    # Check that the plan can be converted back to a matrix
    K2 = DD.plan_to_dense_matrix(P, c, 1, false)
    @test MOT.l1(K, K2) < 1e-8
    # Using both partitions
    K2 = DD.plan_to_dense_matrix(P, c, 2, false)
    @test MOT.l1(K, K2) < 1e-8

    # And also using the SparseMatrix method
    K_sparse = sparse(K)
    droptol!(K_sparse, 1e-10)
    K2 = DD.plan_to_sparse_matrix(P, c, false, 1e-10)
    @test MOT.l1(K_sparse, K2) < 1e-8

    # The other direction we will check when we show `solvecell!` 
    # and `iterate` work.    
end
