# DomDecOT.jl

Domain decomposition algorithm for entropic optimal transport (https://arxiv.org/abs/2001.10986), implemented in Julia.

# Examples
Jupyter notebooks are available in the separate repo https://github.com/ismedina/DomDecOTExamples.jl. 

# Documentation

For documentation see https://ismedina.github.io/DomDecOT.jl/dev/. 

# Installation

This package is built on top of `MultiScaleOT.jl`, which provides most of the type interface and Sinkhorn solvers. If you want to use it run
```julia-repl
] add https://github.com/ismedina/MultiScaleOT.jl
```
to add the `MultiScaleOT.jl` library, and then 
```julia-repl
] add https://github.com/ismedina/DomDecOT.jl
```
to use this one. 