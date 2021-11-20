using Test
using TestSetExtensions

# Install MultiScaleOT
#Pkg.add(Pkg.PackageSpec(url="https://github.com/ismedina/MultiScaleOT.jl"))

@testset "All the tests" begin
    @includetests ARGS
end
