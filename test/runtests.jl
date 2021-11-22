using Test
using TestSetExtensions
import Pkg

# Install MultiScaleOT (for CI)
Pkg.add(Pkg.PackageSpec(url="https://github.com/ismedina/MultiScaleOT.jl"))

@testset "All the tests" begin
    @includetests ARGS
end
