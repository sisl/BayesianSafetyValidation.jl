function coverage(gp, models; m=[20,20], dist=(z1,z2)->norm(z1 - z2, 2))
    n = prod(m)
    a₁, b₁ = models[1].range[1], models[1].range[end]
    a₂, b₂ = models[2].range[1], models[2].range[end]
    θ₁ = range(a₁, b₁, length=m[1])
    θ₂ = range(a₂, b₂, length=m[2])
    δ₁ = θ₁[2] - θ₁[1]
    δ₂ = θ₂[2] - θ₂[1]
    δ = (δ₁ + δ₂)/2
    grid = [[x1,x2] for x1 in θ₁, x2 in θ₂]
    d = (j,X) -> minimum([dist([x...], grid[j]) for x in X])
    X = collect(eachcol(gp.x))
    return 1 - (1/δ) * sum(min(d(j,X), δ) / n for j in 1:n)
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


function region_characterization(gp, models, sparams; m=[200,200], failures_only=false)
    a₁, b₁ = models[1].range[1], models[1].range[end]
    a₂, b₂ = models[2].range[1], models[2].range[end]
    θ₁ = range(a₁, b₁, length=m[1])
    θ₂ = range(a₂, b₂, length=m[2])

    X = [[x1,x2] for x1 in θ₁, x2 in θ₂]
    X = reshape(X, (:,))
    Y_true = System.evaluate(sparams, X; verbose=false) .>= 0.5 # Important for probabilistic-valued systems.
    if failures_only
        condition = (ŷ, y) -> y == 1 && ŷ == 1 # compare only failures
    else
        condition = (ŷ, y) -> y == ŷ # compare all classifications
    end

    return mean(condition(g_gp(gp, x), Y_true[i]) for (i,x) in enumerate(X))
end


function compute_metrics(gp, models, sparams; compute_truth=true, relative_error=true)
    est = p_estimate(gp, models)

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
