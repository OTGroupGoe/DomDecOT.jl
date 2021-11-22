# DomDecOT.jl

Domain decomposition algorithm for entropic optimal transport (https://arxiv.org/abs/2001.10986), implemented in Julia.

For documentation see https://ismedina.github.io/DomDecOT.jl/dev/.

# Installation

This package is built on top of `MultiScaleOT.jl`, which provides most of the type interface and Sinkhorn solvers. You will need to run: 
```julia-repl
] add MultiScaleOT
```
to add the `MultiScaleOT.jl` library, and then 
```julia-repl
] add DomDecOT
```
to use this one. 

# Examples
Both Jupyter and Pluto notebooks are available in the `notebooks` and `pluto-notebooks` folder. You can inspect an `.html` generated from these files online. Alternatively, you can run them locally or the cloud.

## Running the examples locally

Open Julia in the notebooks folder and run 
```julia-repl
] instantiate
```
to get the environment for the package set up (just the first time). With this the notebooks should download and install the `MultiScaleOT` and `DomDecOT` libraries.


## Run them online

Jupyter notebooks can be run online on Binder; you just need to click below (please allow for some time until the Binder server starts)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ismedina/DomDecOT.jl/HEAD)
