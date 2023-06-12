"""
p(fail) estimate across uniform space of the Gaussian process.
"""
function p_estimate(gp, models; num_steps=500, m=fill(num_steps, length(models)), grid=true)
    if grid
        X = make_broadcastable_grid(models, m)
        w = broadcast((x...)->prod([pdf(model.distribution, xx) for (xx, model) in zip(x, models)]), X...)
    else
        U = [Uniform(model.range[1], model.range[end]) for model in models]
        X = [rand(u, n) for (u, n) in zip(U, m)]
        X = make_broadcastable_grid(X)
        w = broadcast((x...)->prod([pdf(model.distribution, xx) / pdf(u, xx) for (xx, model, u) in zip(x, models, U)]), X...)
    end
    w = reshape(w, :)
    Y = broadcast((x...)->g_gp(gp, [x...]), X...)
    Y = reshape(Y, :)
    if grid
        return w'Y / sum(w)
    else
        return 1/length(w) * sum(w[i] * Y[i] for i in eachindex(w))
    end
end

"""
Return mean and variance of `N` different importance sampling estimates of p(fail).
"""
function lw_statistics(gp, models; N=1, m=fill(500, length(models)))
    lw_ests = [p_estimate(gp, models; m, grid=false) for _ in 1:N]
    μ = mean(lw_ests)
    σ² = var(lw_ests)
    return μ, σ²
end


"""
Biased p(fail) estimate across discrete uniform space of the Gaussian process.
"""
function p_estimate_biased(gp, models; m=fill(500, length(models)))
    θ₁ = range(models[1].range[1], models[1].range[end], length=m[1])
    θ₂ = range(models[2].range[1], models[2].range[end], length=m[2])
    Y = [g_gp(gp, [x1,x2]) for x1 in θ₁ for x2 in θ₂]
    return mean(Y)
end


"""
p(fail) estimate across gridded space of the Gaussian process without using the operational likelihood model (i.e., this is failure rate over the GP output across the domain).
"""
function p_estimate_naive(gp, models; m=fill(500, length(models)), grid=true)
    if grid
        X = make_broadcastable_grid(models, m)
    else
        U = [Uniform(model.range[1], model.range[end]) for model in models]
        X = [rand(u, n) for (u, n) in zip(U, m)]
        X = make_broadcastable_grid(X)
    end
    Y = broadcast((x...)->g_gp(gp, [x...]), X...)
    return mean(Y)
end


# """
# p(fail) estimate across sampled true observations.
# """
# function p_estimate_samples(gp, models)
#     Y = inverse_logit.(gp.y)
#     θ₁ = gp.x[1,:]
#     θ₂ = gp.x[2,:]
#     w = [pdf(models[1].distribution, x1)*pdf(models[2].distribution, x2) for (x1,x2) in zip(θ₁,θ₂)]
#     # mean(w .* Y)
#     return w'Y / sum(w)
# end


# function likelihood_weights(gp, models)
#     samples = eachcol(gp.x)
#     w = [prod(pdf(models[i].distribution, x) for (i,x) in enumerate(sample)) for sample in samples]
#     return w
# end


# """
# p(fail) estimate using the leave-one-out likelihood from the discrete GP points.
# """
# function p_estimate_loo(gp, models)
#     Y = gp.y
#     w_p = likelihood_weights(gp, models)
#     w_q = proposal_likelihood(gp)
#     w = w_p ./ w_q
#     return mean(w .* Y)
#     # return mean((w_p[i]/w_q[i]) * Y[i] for i in 1:length(Y))
# end


# function proposal_likelihood(gp)
#     y = gp.y
#     μ, σ2 = GaussianProcesses.predict_LOO(gp)
#     return [exp(Distributions.logpdf(Normal(μi,√σi2), yi)) for (μi,σi2,yi) in zip(μ,σ2,y)]
# end
