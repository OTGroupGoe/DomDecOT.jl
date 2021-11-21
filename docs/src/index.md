# Home

## The domain decomposition algorithm

`DomDecOT.jl` is a Julia library for solving (entropic) optimal transport (OT) with the [domain decomposition algorithm for optimal transport](https://arxiv.org/abs/2001.10986). The domain decomposition methods works by dividing the optimal transport problem into smaller problems, solving them independently and combining the partial solutions together. Under an appropriate choice of the subproblems, it converges linearly to the optimal solution. Thus it is amenable to parallelization and efficient.

`DomDecOT.jl` is designed to be flexible, memory-savy and type-stable. The main routines come in serial and parallel flavors, allowing to leverage multiple cores for faster computation. Types can be extended for custom forms of representing measures. The library is built on top of `MultiScaleOT.jl`, which provides types for measures, multi-scale measures and many utils.
