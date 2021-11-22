# TODO: Test
"""
    DEFAULT_PARAMETERS

Default parameters for the domdec routines
"""
const DEFAULT_PARAMETERS = (;
    epsilon = 1.0,
    solver_max_error = 1e-6,
    solver_max_error_rel = true,
    solver_max_iter = 10000,
    solver_verbose = true,
    solver_truncation = 1e-15, # In case you are using sparse sinkhorn
    balance = true, 
    truncate = true, 
    truncate_Ythresh = 1e-15,
    truncate_Ythresh_rel = true,
    parallel_iteration = false
) 

"""
    default_domdec_eps_schedule(depth::Int, target_eps; Nsteps = 3, factor = 2., last_iter = Float64[])

Return schedules for the layer, epsilon and number of domdec iterations. 
"""
function default_domdec_eps_schedule(depth::Int, target_eps; Nsteps = 3, factor = 2., last_iter = Float64[])
    eps_schedule = MultiScaleOT.scaling_schedule(depth, 
                                                Float64(target_eps), 
                                                Nsteps, 
                                                Float64(factor); 
                                                last_iter)

    layer_schedule = MultiScaleOT.template_schedule(depth, 
                                                    fill(1, Nsteps), 
                                                    collect(1:depth); 
                                                    last_iter = fill(depth, length(last_iter)))
    # How many domain decomposition iterations per epsilon
    iters_schedule = MultiScaleOT.template_schedule(depth, 
                                                    [4, 2, 2], 
                                                    ones(Int, depth); 
                                                    last_iter = fill(2, length(last_iter)))
    layer_schedule, eps_schedule, iters_schedule
end

"""
    make_domdec_schedule(; nt...) 

Build schedules for all the parameters in `DomDecOT.DEFAULT_PARAMETERS`, 
as well as those given in the NamedTuple `nt`.
"""
function make_domdec_schedule(; nt...) 
    # Assert that all parameters not given by default are in the schedule
    (:epsilon ∉ keys(nt)) && error("must provide epsilon schedule")
    (:layer ∉ keys(nt)) && error("must provide layer schedule")
    (:domdec_iters ∉ keys(nt)) && error("must provide domdec_iters schedule")
    # Add from DEFAULT_PARAMETERS the ones that might be missing
    nt = (; DEFAULT_PARAMETERS..., nt...)
    MultiScaleOT.make_schedule(; nt...) 
end