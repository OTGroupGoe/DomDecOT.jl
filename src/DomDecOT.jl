module DomDecOT

import Base: show
import LinearAlgebra: mul!, dot
using SparseArrays
import MultiScaleOT
import MultiScaleOT: AbstractMeasure, 
            GridMeasure, 
            CloudMeasure,
            MultiScaleMeasure,
            npoints, 
            get_kernel,
            get_kernel!, 
            KL,
            sinkhorn_stabilized!, 
            log_sinkhorn!

include("cells.jl")

include("domdecplan.jl")
export DomDecPlan,
    plan_to_dense_matrix,
    plan_to_sparse_matrix

include("iteration.jl")
export iterate!,
    iterate_serial!,
    iterate_parallel!

include("scores.jl")
export primal_score, 
    dual_score,
    PD_gap

include("duals.jl")
export smooth_alpha_field,
    smooth_alpha_and_beta_fields

include("balance.jl")

include("sinkhorn.jl")
export domdec_sinkhorn_stabilized!, 
    domdec_logsinkhorn!, 
    domdec_sinkhorn_autofix_log!

include("multiscale.jl")
export refine_plan

include("display.jl")

include("parameters.jl")
export default_domdec_eps_schedule, 
    make_domdec_schedule

# Precompile routines
# include("precompile/precompile.jl")
end # module
