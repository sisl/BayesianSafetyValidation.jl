function coverage_vec(gp, models; num_steps=20, m=fill(num_steps, length(models)), dist=(z1,z2)->norm(z1 - z2, 2))
    n = prod(m)
    ranges = [range(model.range[1], model.range[end], length=l) for (model, l) in zip(models, m)]
    inputs = [Vector(r) for r in ranges]
    δ = sum(range[2] - range[1] for range in ranges) / length(ranges)
    grid = make_broadcastable_grid(inputs)
    # TODO this will materialize the full input grid.  can probably avoid with some clever broadcasting
    grid = broadcast((x...)->[x...], grid...)
    d = (j,X) -> minimum([dist([x...], grid[j]) for x in X])
    X = collect(eachcol(gp.x))
    return [1 - min(d(j, X), δ) / δ for j in 1:n]
end

function coverage(gp, models; num_steps=20, m=fill(num_steps, length(models)), dist=(z1,z2)->norm(z1 - z2, 2))
    coverages = coverage_vec(gp, models; num_steps, m, dist)
    return sum(coverages) / size(coverages)[1]
end

function quantile_coverage(gp, models, p; num_steps=20, m=fill(num_steps, length(models)), dist=(z1,z2)->norm(z1 - z2, 2))
    coverages = coverage_vec(gp, models; num_steps, m, dist)
    return quantile(coverages, p)
end

"""
Return most-likely failure.
"""
most_likely_failure(gp, models; kwargs...) = most_likely_failure(gp.x, gp.y, models; kwargs...)
most_likely_failure(x, y, models; kwargs...) = vec(most_likely_failures(x, y, models, 1; kwargs...))
most_likely_failures(gp, models, n::Int=1; kwargs...) = most_likely_failures(gp.x, gp.y, models, n; kwargs...)
function most_likely_failures(x, y, models, n::Int=1; return_index=false)
    likelihoods = [pdf(models, x[:,i]) * (y[i] >= 0) for i in eachindex(y)]
    top_n_idx = partialsortperm(likelihoods, 1:n, rev=true)
    if return_index
        return x[:, top_n_idx], top_n_idx
    else
        return x[:, top_n_idx]
    end
end


most_likely_failure_likelihood(gp, models) = maximum([pdf(models, gp.x[:,i]) * (gp.y[i] > 0) for i in eachindex(gp.y)])


"""
Return all failures.
"""
falsification(gp) = falsification(gp.x, gp.y)
falsification(x, y) = x[:, y .>= 0]


function region_characterization(gp, models, sparams; num_steps=200, m=fill(num_steps, length(models)), failures_only=false)
    X = make_broadcastable_grid(models, m)
    f(x...) = System.evaluate(sparams, [[x...]])[1]
    Y_true = f.(X...) .>= 0.5 # Important for probabilistic-valued functions.
    Y_pred = gp_output(gp, models; m, f=g_gp)
    if failures_only
        condition = (ŷ, y) -> y == 1 && ŷ == 1 # compare only failures
    else
        condition = (ŷ, y) -> y == ŷ # compare all classifications
    end

    return mean(condition.(Y_true, Y_pred))
end


function compute_metrics(gp, models, sparams; weights=missing, compute_truth=false, relative_error=true)
    if ismissing(weights)
        est = p_estimate(gp, models)
    else
        est = is_self_normalizing(gp, weights)
    end

    if compute_truth
        truth = truth_estimate(sparams, models)
        gp_error = abs(est - truth)
        @info "p(fail) estimate: $est"
        if relative_error
            gp_error = gp_error / truth
            @info "p(fail) relative error: $gp_error"
        else
            @info "p(fail) error: $gp_error"
        end
    else
        @info "p(fail) estimate: $est"
    end

    num_failures = sum(gp.y .>= 0) # logits
    failure_rate = num_failures / length(gp.y)
    @info "Falsification = $num_failures ($failure_rate)"

    ℓ_most_likely_failure = most_likely_failure_likelihood(gp, models)
    @info "MLFA = $ℓ_most_likely_failure"

    coverage_metric = coverage(gp, models)
    @info "Coverage = $coverage_metric"

    if compute_truth
        region_metric = region_characterization(gp, models, sparams)
        @info "Region characterization = $region_metric"
    end
end
