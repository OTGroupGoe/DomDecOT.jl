# Home

## The domain decomposition algorithm

Domain decomposition algorithm is an efficient and parallelizable algorithm for (entropic) optimal transport. It works by dividing the optimal transport problem into smaller problems, solving them independently and combining the partial solutions together. Under an appropriate choice of the subproblems, linear convergence to the optimal solution is proven [ref].

`DomDecOT.jl` is a Julia library that implements the domain decomposition algorithm. It is designed to be flexible, memory-savy and type-stable. The main routines come in serial and parallel flavors, allowing to leverage multiple cores for faster computation.

`DomDecOT.jl` builds upon the library `MultiScaleOT.jl`, which might be considered a Julia version of the python library with the same name.

was introduced by Benamou [ref] for continuous optimal transport, and then revisited by Bonafini and Schmitzer [ref] for entropic, discrete optimal transport. It is a 

# Manual



