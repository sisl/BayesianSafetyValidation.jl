"""
Baseline: `N` discrete points spanning grid defined by `models`.
"""
function baseline_discrete(models, N=nothing; m=nothing)
    if isnothing(N) && !isnothing(m)
        mx, my = m
    else
        mx = my = ceil(Int, sqrt(N))
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
	x = Sobol.next!(sobol)
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
function baseline_lhc(models, N; gens=100)
    lhc_plan, _ = LHCoptim(N, length(models), gens)
    min = map(m->first(m.range), models)
    max = map(m->last(m.range), models)
    X = Matrix(scaleLHC(lhc_plan, [(min[1], max[1]), (min[2], max[2])])')
    return X
end


"""
Run baseline function `baseline` evaluated across function `f`.
"""
function run_baseline(f, baseline, models, N; gp_args)
    X = baseline(models, N)
    Y = f(collect(eachcol(X))) # run all at once as Vector of inputs
    gp = gp_fit(X, Y; gp_args...)
    return gp
end


"""
Run all baseline functions evaluated across `f`.
"""
function run_baselines(gp, sparams, models, N; is=false, copmute_truth=true, show_truth=true, show_plots=true, input_discretization_steps=500, show_data=true, data_ms=4, use_circles=false, soft=true, gp_args=DEFAULT_GP_ARGS, tight=false)
    baselines = Dict{Any, Any}(
        "discrete"=>baseline_discrete,
        "uniform"=>baseline_uniform,
        "sobol"=>baseline_sobol,
        "lhc"=>baseline_lhc,
    )
    f = x->begin
        System.reset(sparams)
        inputs = []
        for xi in x
            xi = [xi...]
            input = System.generate_input(sparams, xi; models)
            push!(inputs, input)
        end
        System.evaluate(sparams, inputs)
    end
    if show_truth
        truth = truth_estimate(sparams, models, num_steps=input_discretization_steps)
        @info "truth est.: $truth"
    else
        truth = 0
    end
    for (k,v) in baselines
        @info "Running baseline $k"
        baseline = v
        baseline_gp = run_baseline(f, baseline, models, N; gp_args)
        # baseline_gp = @suppress run_baseline(f, baseline, models, N; gp_args)
        baselines[k] = baseline_gp # overwrite function with Gaussian procces object.
    end

    if copmute_truth
        errors, estimates, num_failures, failure_rates, ℓ_most_likely_failures, coverage_metrics, region_metrics = test_baselines(baselines, models, sparams, truth; input_discretization_steps)
    end

    if !isnothing(gp) && copmute_truth
        est = is ? is_estimate_q(gp, models) : p_estimate(gp, models, num_steps=input_discretization_steps)[1]
        gp_error = est - truth
        gp_num_failures = sum(gp.y .>= 0) # logits
        gp_failure_rate = gp_num_failures / length(gp.y)
        gp_ℓ_most_likely_failure = most_likely_failure_likelihood(gp, models)
        gp_coverage_metric = coverage(gp, models; num_steps=input_discretization_steps)
        gp_region_metric = region_characterization(gp, models, sparams; num_steps=input_discretization_steps)
        @info "GP: $gp_error"
        errors["GP"] = gp_error
        estimates["GP"] = est
        num_failures["GP"] = gp_num_failures
        failure_rates["GP"] = gp_failure_rate
        ℓ_most_likely_failures["GP"] = gp_ℓ_most_likely_failure
        coverage_metrics["GP"] = gp_coverage_metric
        region_metrics["GP"] = gp_region_metric
    else
        errors = estimates = num_failures = failure_rates = ℓ_most_likely_failures = coverage_metrics = region_metrics = nothing
    end

    if show_truth && copmute_truth
        display(sort(errors, lt=(x1,x2)->isless(abs(errors[x1]), abs(errors[x2])))) # absolute errors sorted
    end
    if show_plots
        plot_baselines(gp, sparams, baselines, models; show_truth, show_data, data_ms, use_circles, soft, tight) |> display
    end
    return baselines, errors, estimates, num_failures, failure_rates, ℓ_most_likely_failures, coverage_metrics, region_metrics
end


"""
Get p(fail) estimate for each baseline.
"""
function test_baselines(baselines::Dict, models, sparams, truth=0; is=false, input_discretization_steps=500)
    errors = Dict()
    estimates = Dict()
    num_of_failures = Dict()
    failure_rates = Dict()
    ℓ_most_likely_failures = Dict()
    coverage_metrics = Dict()
    region_metrics = Dict()
    for (k,v) in baselines
        est = is ? is_estimate_q(v, models) : p_estimate(v, models; num_steps=input_discretization_steps)[1]
        err = est - truth
        num_failures = sum(v.y .>= 0) # logits
        failure_rate = num_failures / length(v.y)
        ℓ_most_likely_failure = most_likely_failure_likelihood(v, models)
        coverage_metric = coverage(v, models; num_steps=input_discretization_steps)
        region_metric = region_characterization(v, models, sparams; num_steps=input_discretization_steps)

        @info "$k: $err"
        errors[k] = err
        estimates[k] = est
        num_of_failures[k] = num_failures
        failure_rates[k] = failure_rate
        ℓ_most_likely_failures[k] = ℓ_most_likely_failure
        coverage_metrics[k] = coverage_metric
        region_metrics[k] = region_metric
    end
    return errors, estimates, num_of_failures, failure_rates, ℓ_most_likely_failures, coverage_metrics, region_metrics
end


"""
Use discrete grid to sweep function `f` to get 'truth' estimate of p(fail).
"""
function truth_estimate(sparams, models; num_steps=500, m=fill(num_steps, length(models)), return_mean=false)
    p(x) = prod(pdf(models[i].distribution, x[i]) for i in eachindex(models))
    X = make_broadcastable_grid(models, m)
    # we could make a single call to System.evaluate, but that would
    # require materializing the full input grid.  so instead we use
    # broadcasting (along with some extra array wrapping and
    # unwrapping to match the System.evaluate interface)
    f(x...) = System.evaluate(sparams, [[x...]])[1]
    Y = @suppress f.(X...)
    Y = Y .>= 0.5 # Important for probabilistic-valued functions.
    Y = reshape(Y, :)
    if return_mean
        return mean(Y)
    else
        g(x...) = p([x...])
        w = g.(X...)
        w = reshape(w, :)
        return w'Y / sum(w)
    end
end



function plot_baselines(gp, sparams, baselines, models; show_truth=true, show_data=true, data_ms=4, use_circles=false, soft=true, tight=false, titlefontsize=24)
    plots = []
    rd = 6
    casing = Dict(
        "lhc"=>"LHC",
        "sobol"=>"Sobol",
        "discrete"=>"Discrete",
        "uniform"=>"Uniform$(all(isa(model.distribution, Distributions.Uniform) for model in models) ? " (Nominal)" : "")",
    )
    for (k,v) in baselines
        if soft
            plt = plot_soft_boundary(v, models; show_data, tight, ms=data_ms, use_circles)
        else
            plt = plot_hard_boundary(v, models; show_data, tight, ms=data_ms, use_circles)
        end
        plot!(; title=casing[k], titlefontsize, colorbar=false)
        push!(plots, plt)
    end
    if soft
        plot_soft_boundary(gp, models; show_data, tight, ms=data_ms, use_circles)
    else
        plot_hard_boundary(gp, models; show_data, tight, ms=data_ms, use_circles)
    end
    gp_title = "Bayesian safety validation"
    plt = plot!(; title=gp_title, titlefontsize, colorbar=false)
    push!(plots, plt)

    if show_truth
        plot_truth(sparams, models; hard=!soft, tight, edges=true)
        plt = plot!(; title="Truth", titlefontsize, colorbar=false)
        push!(plots, plt)
    end

    if tight
        size = (450*length(plots), 460)
    else
        size = (450*length(plots), 400)
    end
    plot(plots...; layout=(1,length(plots)), size, margin=5Plots.mm)
end
