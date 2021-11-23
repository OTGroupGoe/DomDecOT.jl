# DomDecOT.jl

Domain decomposition algorithm for entropic optimal transport (https://arxiv.org/abs/2001.10986), implemented in Julia.

For documentation see https://ismedina.github.io/DomDecOT.jl/dev/. 

# Examples
It is recommended to take a look at the examples before trying the library. Jupyter notebooks are available in the `notebooks`  folder. You can inspect an `.html` generated from these files online. Alternatively, you can run them locally.

## Running the examples locally

Clone the repo, open Julia in the notebooks folder and run 
```julia-repl
import Pkg; Pkg.activate("."); Pkg.instantiate()
```
to get the environment for the package set up (just the first time). This will also download and install the `MultiScaleOT` and `DomDecOT` libraries.

# Installation (using the package outside the repo)

This package is built on top of `MultiScaleOT.jl`, which provides most of the type interface and Sinkhorn solvers. If you want to use it run
```julia-repl
] add https://github.com/ismedina/MultiScaleOT.jl
```
to add the `MultiScaleOT.jl` library, and then 
```julia-repl
] add https://github.com/ismedina/DomDecOT.jl
```
to use this one. 
