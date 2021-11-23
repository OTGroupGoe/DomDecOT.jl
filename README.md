# DomDecOT.jl

Domain decomposition algorithm for entropic optimal transport (https://arxiv.org/abs/2001.10986), implemented in Julia.

For documentation see https://ismedina.github.io/DomDecOT.jl/dev/.

# Installation

This package is built on top of `MultiScaleOT.jl`, which provides most of the type interface and Sinkhorn solvers. You will need to run in the Julia prompt: 
```julia-repl
] add https://github.com/ismedina/MultiScaleOT.jl
```
to add the `MultiScaleOT.jl` library, and then 
```julia-repl
] add https://github.com/ismedina/DomDecOT.jl
```
to use this one. 

# Examples
Both Jupyter and Pluto notebooks are available in the `notebooks` and `pluto-notebooks` folder. You can inspect an `.html` generated from these files online. Alternatively, you can run them online.

## Running the examples locally

Open Julia in the notebooks folder and run 
```julia-repl
import Pkg; Pkg.activate("."); Pkg.instantiate("")
```
to get the environment for the package set up (just the first time). This will also download and install the `MultiScaleOT` and `DomDecOT` libraries.