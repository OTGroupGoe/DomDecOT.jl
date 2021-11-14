# CODE STATUS: TESTED, REVISED
# TODO: Lots of code still in .sinkhorn
using StatsBase: mean
using SparseArrays
import MultiScaleOT as MOT
import MultiScaleOT: sinkhorn!, sinkhorn_stabilized!

"""
    domdec_sinkhorn_stabilized!(β, α, νJ, νI, μJ, C, ε; kwargs...)

Solve cell problem using the stabilized Sinkhorn algorithm of the 
`MultiScaleOT` library. 

# Arguments
* β: initial Y dual potential
* α: initial X dual potential
* νJ: Y cell marginal 
* νI: global Y marginal supported on the same points as νJ_global
* μJ: X cell marginal
* C: cost matrix. It is transformed inplace to yield the primal plan 
* ε: regularization
"""
    
function domdec_sinkhorn_stabilized!(β, α, νJ, νI, μJ, C, ε; 
            max_iter = 1000, max_error = 1e-8, 
            max_error_rel=true, verbose = true)
    # Rename cost to K for code clarity
    K = C
    get_kernel!(K, β, α, νI, μJ, ε)
    status = sinkhorn_stabilized!(β, α, νJ, μJ, K, ε; 
                max_iter, max_error, max_error_rel, verbose)
    return status
end    

# TODO: provide auto-fix 
# TODO: provide sparsesinkhorn 
# TODO: provide log-sinkhorn