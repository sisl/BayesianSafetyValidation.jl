"""
Fit Gaussian process surrogate model to results.
"""
function gp_fit(X, Y; ν=5/2, ll=1.0, lσ=0.5, opt=false, mean=0.5)
    kernel = Matern(ν, ll, lσ)
    mean_f = MeanConst(mean)
    gp = GP(X, Y, mean_f, kernel)
    opt && @suppress optimize!(gp, NelderMead())
    return gp
end

function gp_fit(results::Dict)
    X::Matrix = cmat([[k...] for k in collect(keys(results))])
    Y::Vector{Bool} = collect(values(results))
    return gp_fit(X, Y)
end


"""
Predicted GP mean and covariance, given as a vector (reshaped to a matrix)
"""
predict_f_vec = (gp,x)->predict_f(gp, reshape(x', (2,1)))


"""
Predicted GP mean (`predict_f` outputs a [[mean], [cov]] so we want just the mean as [1][1])
"""
f_gp = (gp,x)->predict_f_vec(gp,x)[1][1]


"""
Predicted GP variance (`predict_f` outputs a [[mean], [cov]] so we want just the variance as [2][1])
"""
σ²_gp = (gp,x)->predict_f_vec(gp,x)[2][1]


"""
Run the GP and get the predicted output across a discretized space defined by `m[i]` points between the model ranges.

**TODO**: generalize to more than 2 dimensions (NOTE: [x,y] and the "for" ordering of `y` then `x`. This is to make sure the matrix is in the same orientation for plotting with the smallest values at the bottom-left origin.)
"""
function gp_output(gp, models::Vector{OperationalParameters}; m=fill(200, length(models)), f=f_gp, clamping=true)
    p(x) = prod(m->pdf(m.distribution,x), models)
    y = [f(gp, [x,y]) for y in range(models[2].range[1], models[2].range[end], length=m[2]), x in range(models[1].range[1], models[1].range[end], length=m[1])]

    if clamping && !isempty(gp.y)
        # y = clamp.(y, 0, 1)
        if !all(y .== 0)
            y = sigmoid.(y)
            y = normalize01(y)
        end
    end
    return y
end


"""
Upper confidence bound. Note, strictly greater than.
"""
ucb(gp, x; λ=1, hard=false) = (hard ? f_gp(gp,x) > 0.5 : f_gp(gp,x)) + λ*σ²_gp(gp,x)


"""
Lower confidence bound. Note, strictly greater than.
"""
lcb(gp, x; λ=1, hard=false) = (hard ? f_gp(gp,x) > 0.5 : f_gp(gp,x)) - λ*σ²_gp(gp,x)


"""
Gaussian process acquisition function to determine which point to sample next to reduce uncertainty over entire space (using variance or stdev in surrogate).
"""
function uncertainty_acquisition(gp, x, models::Vector{OperationalParameters}; λ=1, t=1, var=false)
    μ_σ² = predict_f_vec(gp, x)
    σ² = μ_σ²[2][1]
    if var
        return λ*σ²
    else
        σ = sqrt(σ²)
        return λ*σ
    end
end


"""
Gaussian process acquisition function to determine which point to sample next to reduce uncertainty around decision boundary.
"""
function boundary_acquisition(gp, x, models::Vector{OperationalParameters}; λ=1, t=1, include_p=true, include_decay=true)
	μ_σ² = predict_f_vec(gp, x)
	μ = μ_σ²[1][1]
	σ = sqrt(μ_σ²[2][1])
	μ′ = μ * (1 - μ)
    p = prod(pdf(models[i].distribution, x[i]) for i in eachindex(models))

    if include_p
        if !include_decay
            t = 1 # overwrite
        end
        acquisition = (μ′ + λ*σ) * p^(1/t)
    else
        acquisition = μ′ + λ*σ
    end
	return acquisition
end


"""
Gaussian process acquisition function to determine which point to sample next to reduce uncertainty of failure distribution.
"""
function failure_region_acquisition(gp, x, models::Vector{OperationalParameters}; λ=1, t=1)
	μ_σ² = predict_f_vec(gp, x)
	μ = μ_σ²[1][1]
	σ² = μ_σ²[2][1]
	σ = sqrt(σ²)
    p = prod(pdf(models[i].distribution, x[i]) for i in eachindex(models))

    f̂ = (μ + λ*σ) > 0.5 # UCB
    acquisition = f̂ * p

	return acquisition
end


"""
Combining multi-objective acquisition functions.
"""
function multi_objective_acqusition(gp, x, models; λ=1)
    return failure_region_acquisition(gp, x, models; λ) * boundary_acquisition(gp, x, models; λ) * uncertainty_acquisition(gp, x, models; λ)
end


"""
Get the next recommended sample point based on the operational models and failure boundary.
"""
function get_next_point(gp, models; acq, acq_explore=nothing)
    model_ranges = get_model_ranges(models)
    y_gp = gp_output(gp, models; f=acq, clamping=false)
    if !isnothing(acq_explore)
        explore_gp = gp_output(gp, models; f=acq_explore, clamping=false)
        explore_gp = normalize01(explore_gp)
        y_gp = normalize01(y_gp)
        y_gp = explore_gp + y_gp
    end
    next_point = argmax(model_ranges[1], model_ranges[2], y_gp)
    return next_point
end


"""
Stochastically sample next point using normalized weights.
"""
function sample_next_point(gp, models; n=1, r=1, acq, return_weight=false)
    y_gp = gp_output(gp, models; f=acq, clamping=false)
    y_gp = normalize01(y_gp) # to eliminate negative weights
    X = [[x,y] for y in range(models[2].range[1], models[2].range[end], length=size(y_gp,2)), x in range(models[1].range[1], models[1].range[end], length=size(y_gp,1))]
    Z = normalize(y_gp .^ r, 1)
    if all(isnan.(Z))
        Z = ones(size(Z))
    end
    candidate_samples = [X...]
    weights = [Z...]
    indices = eachindex(candidate_samples)
    sampled_indices = sample(indices, StatsBase.Weights(weights), n, replace=true)
    if return_weight
        return candidate_samples[sampled_indices], weights[sampled_indices]
    else
        return candidate_samples[sampled_indices]
    end
end
