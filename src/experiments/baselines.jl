"""
Baseline: `N` discrete points spanning grid defined by `models`.
"""
function baseline_discrete(models, N=nothing; m=nothing)
    if isnothing(N) && !isnothing(m)
        mx, my = m
    else
        mx = my = trunc(Int, sqrt(N))
    end
    model_ranges = get_model_ranges(models, [mx, my])
    X = reduce(hcat, [x; y;;] for x in model_ranges[1] for y in model_ranges[2])
    return X
end


"""
Baseline: Sample `N` random numbers uniformly.
"""
function baseline_uniform(models, N)
    min = map(m->first(m.range), models)
    max = map(m->last(m.range), models)
    U = Product(Uniform.(min, max))
    X = rand(U, N)
    return X
end


"""
Helper to get next number in the Sobol sequence and scale it to a domain.
"""
function next_sobol!(sobol; min=[-5,-5], max=[5,5])
	x = next!(sobol)
	return x .* (max-min) + min
end


"""
Baseline: Sample `N` numbers from the Sobol sequence.
"""
function baseline_sobol(models, N)
    seq = SobolSeq(length(models))
    min = map(m->first(m.range), models)
    max = map(m->last(m.range), models)
    X = reduce(hcat, next_sobol!(seq; min, max) for _ in 1:N)
    return X
end


"""
Baseline: Sample `N` numbers from a Latin hypercube.
"""
function baseline_lhc(models, N; gens=1000)
    lhc_plan, _ = LHCoptim(N, length(models), gens)
    min = map(m->first(m.range), models)
    max = map(m->last(m.range), models)
    X = Matrix(scaleLHC(lhc_plan, [(min[1], max[1]), (min[2], max[2])])')
    return X
end


"""
Run baseline function `baseline` evaluated across function `f`.
"""
function run_baseline(f, baseline, models, N)
    X = baseline(models, N)
    Y = f(collect(eachcol(X))) # run all at once as Vector of inputs
    gp = gp_fit(X, Y)
    return gp
end


"""
Run all baseline functions evaluated across `f`.
"""
function run_baselines(gp, sparams, models, N; is=false)
    baselines = Dict{Any, Any}(
        "discrete"=>baseline_discrete,
        "uniform"=>baseline_uniform,
        "sobol"=>baseline_sobol,
        "lhc"=>baseline_lhc,
    )
    f = x->System.evaluate(sparams, x)
    truth = truth_estimate(sparams, models)
    @info "truth est.: $truth"
    for (k,v) in baselines
        baseline = v
        baseline_gp = @suppress run_baseline(f, baseline, models, N)
        baselines[k] = baseline_gp # overwrite function with Gaussian procces object.
    end
    errors, estimates = test_baselines(baselines, models, truth)
    est = is ? is_estimate_q(gp, models) : p_estimate(gp, models)
    gp_error = est - truth
    @info "GP: $gp_error"
    errors["GP"] = gp_error
    estimates["GP"] = est
    display(sort(errors, lt=(x1,x2)->isless(abs(errors[x1]), abs(errors[x2])))) # absolute errors sorted
    plot_baselines(gp, sparams, baselines, models; errors) |> display
    return baselines, errors, estimates
end


"""
Get p(fail) estimate for each baseline.
"""
function test_baselines(baselines::Dict, models, truth=0; is=false)
    errors = Dict()
    estimates = Dict()
    for (k,v) in baselines
        est = is ? is_estimate_q(v, models) : p_estimate(v, models)
        err = est - truth
        @info "$k: $err"
        errors[k] = err
        estimates[k] = est
    end
    return errors, estimates
end


"""
Use discrete grid to sweep function `f` to get 'truth' estimate of p(fail).
"""
function truth_estimate(sparams, models; m=[500,500])
    p(x) = prod(pdf(models[i].distribution, x[i]) for i in eachindex(models))
    a₁, b₁ = models[1].range[1], models[1].range[end]
    a₂, b₂ = models[2].range[1], models[2].range[end]
    θ₁ = range(a₁, b₁, length=m[1]) # discrete grid
    θ₂ = range(a₂, b₂, length=m[2]) # discrete grid
    Y = @suppress System.evaluate(sparams, [[x,y] for x in θ₁ for y in θ₂])
    w = [p([x,y]) for x in θ₁ for y in θ₂]
    return w'Y / sum(w)
end



function plot_baselines(gp, sparams, baselines, models; errors=nothing)
    plots = []
    rd = 6
    for (k,v) in baselines
        plt = plot_soft_boundary(v, models)
        plot!(title=k, colorbar=false)
        push!(plots, plt)
    end
    plot_soft_boundary(gp, models)
    gp_title = "Gaussian proccess"
    plt = plot!(title=gp_title, colorbar=false)
    push!(plots, plt)

    plot_truth(sparams, models)
    plt = plot!(title="Truth", colorbar=false)
    push!(plots, plt)

    plot(plots..., layout=(1,length(plots)), size=(450*length(plots), 400), margin=5Plots.mm)
end
