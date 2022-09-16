"""
p(fail) estimate across uniform space of the Gaussian process.
"""
function p_estimate(gp, models; m=fill(200, length(models)), grid=true)
    a₁, b₁ = models[1].range[1], models[1].range[end]
    a₂, b₂ = models[2].range[1], models[2].range[end]
    if grid
        θ₁ = range(a₁, b₁, length=m[1]) # discrete grid
        θ₂ = range(a₂, b₂, length=m[2]) # discrete grid
    else
        U₁ = Uniform(a₁, b₁)
        U₂ = Uniform(a₂, b₂)
        θ₁ = rand(U₁, m[1])
        θ₂ = rand(U₂, m[2])
    end
    Y = [f_gp(gp, [x,y]) > 0.5 for x in θ₁ for y in θ₂]
    w = [pdf(models[1].distribution, x)*pdf(models[2].distribution, y) for x in θ₁ for y in θ₂]
    return w'Y / sum(w)
end

"""
Return mean and variance of `N` different importance sampling estimates of p(fail).
"""
function lw_statistics(gp, models; N=1, m=fill(200, length(models)))
    lw_ests = [p_estimate(gp, models; m, grid=false) for _ in 1:N]
    μ = mean(lw_ests)
    σ² = var(lw_ests)
    return μ, σ²
end


"""
Biased p(fail) estimate across uniform space of the Gaussian process.
"""
function p_estimate_biased(gp, models; m=fill(200, length(models)))
    θ₁ = range(models[1].range[1], models[1].range[end], length=m[1])
    θ₂ = range(models[2].range[1], models[2].range[end], length=m[2])
    Y = [f_gp(gp, [x,y]) > 0.5 for x in θ₁ for y in θ₂]
    return mean(Y)
end


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
