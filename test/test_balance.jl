import DomDecOT as DD

@testset ExtendedTestSet "balance_vector" begin
    p0 = [2, 2.]; q0 = [1, 1.]
    p = copy(p0); q = copy(q0)
    @test 0 == DD.balance!(p, q, 0.5, 0.5)
    @test ((p == [1.5, 2]) & (q == [1.5, 1]))

    p = copy(p0); q = copy(q0)
    @test 0 == DD.balance!(p, q, -0.5, 0.5)
    @test ((p == [2.5, 2]) & (q == [0.5, 1]))

    # If more mass than available is demanded, the whole balancing fails
    # TODO: Should "demand more balancing than possible" be an error?
    p = copy(p0); q = copy(q0)
    @test DD.balance!(p, q, 5, 0) == 1
end

@testset ExtendedTestSet "balance_matrix" begin

    Q0 = [2 1
          2 1.]
    Q = copy(Q0)

    μ = [4, 2]
    DD.balance!(Q, μ, 0)
    @test Q == Q0

    μ = [2.0, 4.0]
    DD.balance!(Q, μ, 0)
    @test sum(Q, dims = 1)[:] == μ
    @test sum(Q, dims = 2) == sum(Q0, dims = 2)

    μ = [3, 2]
    Q = copy(Q0)
    #@test_logs (:warn, "Total mass does not equal objective mass") (:warn, "Failed to balance composite cell") DD.balance!(Q, μ, 0)
    @test Q == Q0

    # Random matrix
    n = 10
    Q0 = rand(n, n)
    Q = copy(Q0)
    μ = fill(sum(Q)/n, n)
    DD.balance!(Q, μ, 0)

    @test all(sum(Q, dims = 1)[:] .≈ μ)
    @test all(sum(Q, dims = 2) .≈ sum(Q0, dims = 2))
end
