logit(y; s=1/10) = log(y / (1 - y)) / s
inverse_logit(z; s=1/10) = 1 / (1 + exp(-s*z)) # sigmoid with steepness s

transform(y; ϵ=1e-5) = y*(1 - ϵ) + (1 - y)*ϵ
inverse_transform(ŷ; ϵ=1e-5) = (ŷ - ϵ) / (1 - 2ϵ)

apply(y) = logit(transform(y))
inverse(y) = clamp(inverse_transform(inverse_logit(y)), 0, 1) # small variations based on ϵ may cause GP values just slightly under 0 and slightly over 1, so clamp.

"""
Fit Gaussian process surrogate model to results.
"""
function gp_fit(X, Y; ν=1/2, ll=-0.1, lσ=-0.1, opt=false)
    kernel = Matern(ν, ll, lσ)
    mean_f = MeanZero()
    Z = apply.(Y)
    gp = GP(X, Z, mean_f, kernel)
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
predict_f_vec = (gp,x)->map(y->inverse.(y), predict_f(gp, reshape(x', (:,1))))


"""
Predicted GP mean (`predict_f` outputs a [[mean], [cov]] so we want just the mean as [1][1])
"""
f_gp = (gp,x)->predict_f_vec(gp,x)[1][1]


"""
Predicted GP variance (`predict_f` outputs a [[mean], [cov]] so we want just the variance as [2][1])
"""
σ²_gp = (gp,x)->predict_f_vec(gp,x)[2][1]


"""
Predicted GP failure (hard boundary).
"""
g_gp = (gp,x)->f_gp(gp,x) >= 0.5


function make_broadcastable_grid(models::Vector{OperationalParameters}, m)
    ranges = [range(model.range[1], model.range[end], length=l) for (model, l) in zip(models, m)]
    X = []
    for (i, r) in enumerate(ranges)
        dims = [fill(1, i-1)..., :, fill(1, length(models) - i)...]
        push!(X, reshape(Vector(r), dims...))
    end

    return X
end

function apply_as_list(f, x...)
    return f([x...])
end

"""
Run the GP and get the predicted output across a discretized space defined by `m[i]` points between the model ranges.

**TODO**: generalize to more than 2 dimensions (NOTE: [x,y] and the "for" ordering of `y` then `x`. This is to make sure the matrix is in the same orientation for plotting with the smallest values at the bottom-left origin.)
"""
function gp_output(gp, models::Vector{OperationalParameters}, m=fill(200, length(models)); f=f_gp)
    X = make_broadcastable_grid(models, m)

    g = (x...)->apply_as_list((x...)->f(gp, x...), x...)
    return g.(X...)
    # return [f(gp, [x,y]) for y in range(models[2].range[1], models[2].range[end], length=m[2]), x in range(models[1].range[1], models[1].range[end], length=m[1])]
end


"""
Precompute the operational likelihood over entire design space in grid defined by `m`
"""
function p_output(models::Vector{OperationalParameters}, m=fill(200, length(models)))
    X = make_broadcastable_grid(models, m)

    g = (x...)->apply_as_list((x...)->pdf(models, x...), x...)
    return g.(X...)
end


"""
Upper confidence bound. Note, strictly greater than.
"""
ucb(gp, x; λ=1, hard=false) = (hard ? g_gp(gp,x) : f_gp(gp,x)) + λ*σ²_gp(gp,x)


"""
Lower confidence bound. Note, strictly greater than.
"""
lcb(gp, x; λ=1, hard=false) = (hard ? g_gp(gp,x) : f_gp(gp,x)) - λ*σ²_gp(gp,x)


"""
Gaussian process acquisition function to determine which point to sample next to reduce uncertainty over entire space (using variance or stdev in surrogate).
"""
function uncertainty_acquisition(gp, x, models::Vector{OperationalParameters}; kwargs...)
    μ_σ² = predict_f_vec(gp, x)
    return uncertainty_acquisition(μ_σ²; kwargs...)
end

function uncertainty_acquisition(μ_σ²; var=false)
    σ² = μ_σ²[2][1]
    return var ? σ² : sqrt(σ²)
end
uncertainty_acquisition(μ_σ², p; kwargs...) = uncertainty_acquisition(μ_σ²; kwargs...)


"""
Gaussian process acquisition function to determine which point to sample next to reduce uncertainty around decision boundary.
"""
function boundary_acquisition(gp, x, models::Vector{OperationalParameters}; kwargs...)
	μ_σ² = predict_f_vec(gp, x)
    return boundary_acquisition(μ_σ²; kwargs...)
end

function boundary_acquisition(μ_σ², p; λ=1, t=1, include_p=true, include_decay=true)
	μ = μ_σ²[1][1]
	σ = sqrt(μ_σ²[2][1])
	μ′ = μ * (1 - μ)

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
function failure_region_sampling_acquisition(gp, x, models::Vector{OperationalParameters}; kwargs...)
	μ_σ² = predict_f_vec(gp, x)
    return failure_region_sampling_acquisition(μ_σ²; kwargs...)
end

function failure_region_sampling_acquisition(μ_σ², p; λ=1, probability_valued_system=true)
	μ = μ_σ²[1][1]
	σ² = μ_σ²[2][1]
	σ = sqrt(σ²)

    ĥ = (μ + λ*σ) # UCB
    ĝ = ĥ >= 0.5 # failure indicator
    if probability_valued_system
        acquisition = ĝ * ĥ * p
    else
        acquisition = ĝ * p
    end

	return acquisition
end


"""
Combining multi-objective acquisition functions.
"""
function multi_objective_acqusition(gp, x, models; λ=1)
    return failure_region_sampling_acquisition(gp, x, models; λ) * boundary_acquisition(gp, x, models; λ) * uncertainty_acquisition(gp, x, models; λ)
end


"""
Get the next recommended sample point based on the acquisition function.
"""
function get_next_point(y, F̂, P, models; acq)
    model_ranges = get_model_ranges(models, size(y))
    acq_output = map(acq, F̂, P)
    next_point = [model_ranges[i][x] for (i, x) in enumerate(argmax(acq_output).I)]
    push!(next_point, acq_output[argmax(acq_output)])
    return next_point
end


"""
Stochastically sample next point using normalized weights.
"""
function sample_next_point(y, F̂, P, models; n=1, r=1, acq, return_weight=false)
    acq_output = map(acq, F̂, P)
    acq_output = normalize01(acq_output) # to eliminate negative weights
    # X = [[x,y] for y in range(models[2].range[1], models[2].range[end], length=size(y,2)), x in range(models[1].range[1], models[1].range[end], length=size(y,1))]
    ranges = make_broadcastable_grid(models, size(y))
    X = broadcast((x...)->[x...], ranges...)
    Z = normalize(acq_output .^ r, 1)
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
