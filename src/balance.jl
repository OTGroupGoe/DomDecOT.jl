# CODE STATUS: REVISED, TESTED

"""
    balance!(a, b, δ, threshold; force_balancing = false)

Transfer `δ` mass from vector `a` to `b`. The recipient entries must be
larger than `threshold`; if it is not achieved a warning is thrown.
"""
function balance!(a, b, δ, threshold)
    if δ < 0
        return -balance!(b, a, -δ, threshold)
	end
    len_a = length(a)
    m = 0
    # TODO: add @inbounds and disclaimer in the docstring that no bound
    # checking is performed nor positivity check.
    for i in eachindex(a)
        if b[i] ≥ threshold # Need ≥ so that it still works when threshold==0
            if a[i] < δ
                b[i] += a[i]
                δ -= a[i]
                a[i] = 0
            else
                b[i] += δ;
                a[i] -= δ;
                return 0.0;
            end
        end
    end
    return δ
end

"""
    get_pairwise_delta(d1, d2)

For a pair of values `d1, d2` representing the offset of mass of two vectors, 
computes the maximum possible transfer of mass that does not make any vector
worse than what it is currently. For example, if both vectors have mass in excess
or in defect, it returns 0; if one has in excess and another in defect, it returns
the smallest of these adjustments, with the corresponding sign. 

# Examples
julia> get_pairwise_delta(1,2), get_pairwise_delta(-1, -3)
(0, 0)

julia> get_pairwise_delta(1,-2)
1

julia> get_pairwise_delta(-1,2)
-1
"""
get_pairwise_delta(d1, d2) = min(max(d1,0), max(-d2,0)) - min(max(-d1,0), max(d2,0))

# When the plans have matrix-form
# TODO: allow absolute threshold?
# TODO: the inner nested loop looks improvable, taking one counter `i` to go over
# cells with positive excess and `j` over those with negative. However, since the
# length of the masses will probably be not so large, maybe we do not see much
# performance improvement. Still, worth a try in the future.
"""
    balance!(Q, μ, threshold, force_balancing = true) 

Apply `balance!` with threshold `threshold` on pairs of columns of `Q` 
until each column `Q[:,i]` has mass μ.
When `force_balancing == true`, if a first pass wasn't succesful, 
a second pass is attempted seeting `threshold = 0`.
"""
function balance!(Q::AbstractMatrix, μ, threshold, force_balancing = true)
    Δ = [sum(@views Q[:,i]) - μ[i] for i in eachindex(μ)]
    N = length(μ)
    if N == 1
        return nothing
    end
    if sum(Δ) > 1e-10
        # TODO: should this be an error?
        print("warning: total mass does not equal objective mass")
        # @warn "Total mass does not equal objective mass"
        #return nothing
		# Try to balance even though
    end
    #threshold = 0.0
    for i in 1:N
        for j in i+1:N
            δ0 = get_pairwise_delta(Δ[i], Δ[j])
            #threshold = (δ > 0 ? rel_threshold*μ[j] : rel_threshold*μ[i])
            if abs(δ0) > threshold
                δ = @views balance!(Q[:,i], Q[:,j], δ0, threshold)
                if (δ != 0) & force_balancing
                    # Repeat with zero threshold
                    δ = @views balance!(Q[:,i], Q[:,j], δ, 0.0)
                end
                Δ[i] -= (δ0 - δ)
                Δ[j] += (δ0 - δ)
            end
        end
    end
    if sum(Δ) > 1e-10
        print("warning: balancing failed")
    end
    # (sum(Δ) > 1e-10) && @warn "Failed to balance composite cell" 
    nothing
end