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

function domdec_sinkhorn_autofix!(β, α, νJ, νI, μJ, C, ε; 
            max_iter = 1000, max_error = 1e-8, 
            max_error_rel=true, verbose = true)

    α0 = copy(α)
    β0 = copy(β)
    
    # get kernel in a different variable
    K = get_kernel(C, β, α, νI, μJ, ε)

    status = sinkhorn_stabilized!(β, α, νJ, μJ, K, ε; 
                max_iter, max_error, max_error_rel, verbose)
    if status != 2
        C .= K
        return status
    else
        eps_factor = 2. # Factor to multiply epsilon
        # Upward branch
        eps_doubling_steps = 0
        verbose && print("Increasing epsilon steps:")
        while status == 2
            # Reset dual potentials
            α .= α0
            β .= β0
            ε *= eps_factor
            eps_doubling_steps += 1
            print(" ", eps_doubling_steps)
            K = get_kernel(C, β, α, νI, μJ, ε)
            status = sinkhorn_stabilized!(β, α, νJ, μJ, K, ε; 
                        max_iter, max_error, max_error_rel, verbose)
            if eps_doubling_steps == 100
                status = 1 # Something went wrong
            end
        end
        # Downward branch
        verbose && print("\nDecreasing epsilon steps:")
        for i in 1:eps_doubling_steps
            # Save dual potentials
            α0 .= copy(α)
            β0 .= copy(β)
            ε /= eps_factor
            print(" ", i)
            status = sinkhorn_stabilized!(β, α, νJ, μJ, K, ε; 
                        max_iter, max_error, max_error_rel, verbose)
            if status == 2
                # Give up at this point
                print(" not successful.\n")
                # Get previously succesful duals
                α .= α0
                β .= β0
                ε *= eps_factor
                get_kernel!(C, β, α, νI, μJ, ε) # Update plan
                return status
            end
        end
        # If the downward branch went well, scale and return status
        print(" success!\n")
        get_kernel!(C, β, α, νI, μJ, ε) # Update plan
        return status
    end
end

# TODO: provide sparsesinkhorn 
# TODO: provide log-sinkhorn