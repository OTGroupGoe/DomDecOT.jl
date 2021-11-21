# Library

## Plan

The current state of the domain decomposition algorithm is stored in an `AbstractPlan`. The only `AbstractPlan` provided is called `DomDecPlan`, but new ones can be build if needed. Though it is not strictly necessary to understand its internals to use it, we leave it here for reference: 

```@docs
DomDecPlan
```

It is important to note that a `DomDecPlan` does not store the coupling explicitly. Instead, it stores the marginals of the _basic cells_ of the coupling, from which one can recover the Y-marginals of the partitions where the subproblems are solved. This, together with the dual marginals (that are stored in the fields `alphas` and `betas`) allows to reconstruct the whole coupling if needed, while only using a fraction of the total memory.

## Iteration

```@docs
iterate!
```

Here it is to note that any cost function `c(x,y)` consistent with the columns of `P.mu.points` and `P.nu.points` will work. 

!!! note
    **A note about parameters**

    Parameters are to be packaged in a `NamedTuple`. This is a type-stable analog of a dictionary that can be build as follows

    ```julia-repl
    julia> (; epsilon = 1.0, truncation_thres = 1e-14, verbose = true)
    (epsilon = 1.0, truncation_thres = 1e-14, verbose = true)
    ```

    `DomDecOT.jl` provides easy ways to generate and operate with this tuples, that will be covered in section [ref]


## Solvers

The subproblems are solved with one of the following solvers, that build upon those provided by `MultiScaleOT.jl`.

```@docs
domdec_sinkhorn_stabilized!
```

```@docs
domdec_logsinkhorn!
```

```@docs
domdec_sinkhorn_autofix_log!
```

## Obtaining the coupling and duals

One we have performed a series of iterations, we are usually interested in obtaining the actual coupling.

```@docs 
plan_to_dense_matrix
```

```@docs 
plan_to_sparse_matrix
```

As well as the primal coupling, we are usually interested in obtaining the duals. Note that in a `DomDecPlan` only the duals of each partition are saved, so one must glue them all together to obtain global potentials. This is achieved with the following function.

```@docs
smooth_alpha_and_beta_fields
```

## Computing scores

One can compute the primal and dual scores directly on the `DomDecPlan` by calling the following functions.

```@docs
primal_score
```

```@docs
dual_score
```

```@docs
PD_gap
```

Alternatively, if the global primal and duals are already available in matrix form, one can also use the functions `primal_score_sparse` and `dual_score_sparse` of the `MultiScaleOT.jl` library. [ref]

## Refinement

Domain decomposition becomes really powerful when combined with a hierarchical scheme. For this it is necessary to, from a coarse solution to layer `k-1`, provide a feasible initialization for the finer layer `k`. This can be achieved with `refine_plan`, which also initializes the duals of the new plan by interpolating between those in the old one. 

```@docs
refine_plan
```
