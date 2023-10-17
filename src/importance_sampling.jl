"""
Importance sampling estimate where the q-proposal distribution is uniform.
"""
function is_estimate_uniform(gp, models; m=fill(500, length(models)))
    a₁, b₁ = models[1].range[1], models[1].range[end]
    a₂, b₂ = models[2].range[1], models[2].range[end]
    U₁ = Uniform(a₁, b₁)
    U₂ = Uniform(a₂, b₂)
    θ₁ = rand(U₁, m[1])
    θ₂ = rand(U₂, m[2])
    p = [pdf(models[1].distribution, x)*pdf(models[2].distribution, y) for x in θ₁ for y in θ₂]
    q = [pdf(U₁, x)*pdf(U₂, y) for x in θ₁ for y in θ₂]
    Y = [g_gp(gp, [x,y]) for x in θ₁ for y in θ₂]
    return mean(p ./ q .* Y)
end


"""
Importance sampling estimate where the q-proposal distribution is fit using KDE.
"""
function is_estimate_q(gp, models; failures_only=true, univariate=false)
    X = failures_only ? falsification(gp) : gp.x
    p = x->prod(pdf(models[i].distribution, x[i]) for i in eachindex(models))
    if univariate
        qs = [q_proposal(gp, i; failures_only) for i in eachindex(models)]
        q = x->prod(pdf(qs[i], x[i]) for i in eachindex(models))
        Q = map(x->q(x), eachcol(X))
    else
        q = q_proposal(gp; failures_only)
        Q = map(x->pdf(q, x...), eachcol(X))
    end
    P = map(x->p(x), eachcol(X))
    Y = failures_only ? gp.y[gp.y .== 1] : gp.y
    return mean(P ./ Q .* Y)
end


"""
Return mean and variance of `N` different importance sampling estimates of p(fail).
"""
function is_statistics(gp, models; N=1, m=fill(500, length(models)))
    is_ests = [is_estimate_uniform(gp, models; m) for _ in 1:N]
    μ = mean(is_ests)
    σ² = var(is_ests)
    return μ, σ²
end


"""
Monte Carlo estimate sampling `N` samples from `p`.
"""
function mc_estimte(gp, models; N)
    samples::Vector{Vector} = map(collect, zip(map(m->rand(m.distribution, N), models)...))
    return mean(g_gp(x) for x in samples)
end


"""
Kernel density esimation of the proposal distribution.
"""
q_proposal(gp; failures_only=false) = failures_only ? kde(falsification(gp)') : kde(gp.x')
q_proposal(gp, i; failures_only=false) = failures_only ? kde(falsification(gp)[i,:]) : kde(gp.x[i,:])


"""
Sample from |̂f(x)|p(x).
"""
function is_estimate_theoretically_optimal(gp, models; n=10_000, λ=0.0)
    acq = (gp,x)->failure_region_sampling_acquisition(gp, x, models; λ)
    X, Q = sample_next_point(gp, gp_output(gp, models), models; n, acq, return_weight=true)

    p = x->prod(pdf(models[i].distribution, x[i]) for i in eachindex(models))
    P = map(x->p(x), X)
    Y = map(x->f_gp(gp, x), X)

    return mean(P ./ Q .* Y) # Y already all ones (i.e., all failures)
end


"""
Self-normalzing importance sampling.
"""
function is_self_normalizing(gp, W)
    return sum(inverse(gp.y[i]) * W[i] for i in eachindex(gp.y)) / sum(W)
end