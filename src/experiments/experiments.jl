function run_experiment(sparams, models; T=200, seed=0, record_every=10, show_plots=false, skip_gp=false, save_final_gp=true)
    Random.seed!(seed)
    results = []
    gp = nothing
    samples_to_run = 0
    while true
        if skip_gp
            samples_to_run = min(samples_to_run + 3record_every, 3T)
        else
            @time gp = iteratively_sample(sparams, models;
                seed=seed,
                gp=gp, # continue from last GP
                T=record_every)
            samples_to_run = gp.nobs
        end
        @info "Samples to run: $samples_to_run"

        exp_baselines, exp_errors, exp_estimates, exp_num_failures, exp_failure_rates, exp_ℓ_most_likely_failures, exp_coverage_metrics, exp_region_metrics = run_baselines(gp, sparams, models, samples_to_run; is=false, show_plots)

        # record the [number of data points, error, p(fail) estimate, num failures, failure rate, most-likely failure prob, coverage, region coverage]
        result = merge(vcat, Dict(k=>v.nobs for (k,v) in exp_baselines), exp_errors, exp_estimates, exp_num_failures, exp_failure_rates, exp_ℓ_most_likely_failures, exp_coverage_metrics, exp_region_metrics)
        if skip_gp
            # Use maximum across baselines (ones like the discrete grid truncate to be a square)
            nobs = maximum(v.nobs for (k,v) in exp_baselines)
        else
            nobs = gp.nobs
            result["GP"] = [nobs, exp_errors["GP"], exp_estimates["GP"], exp_num_failures["GP"], exp_failure_rates["GP"], exp_ℓ_most_likely_failures["GP"], exp_coverage_metrics["GP"], exp_region_metrics["GP"]]
        end
        push!(results, result)

        if nobs ≥ 3T
            @info "Finished experiment."
            break
        end
    end
    return results
end


function run_experiments_rng(sparams, models, num_seeds=3; T=200, record_every=10, show_plots=false, skip_gp=false)
    results_set = []
    for seed in 0:num_seeds-1
        results = run_experiment(sparams, models; seed, T, record_every, show_plots, skip_gp)
        push!(results_set, results)
    end
    cr = combine_results(results_set)
    if show_plots
        display(plot_experiment_estimates(cr, sparams, models; is_combined=true))
        display(plot_experiment_error(cr; is_combined=true))
    end
    alert("Experiments finished!")
    return cr
end


function combine_results(results_set::Vector)
    @assert length(unique(map(res->length(res), results_set))) == 1
    combined_results = Vector(undef, length(results_set[1]))
    names = keys(results_set[1][1])
    for i in eachindex(results_set[1])
        combined_results[i] = Dict()
        for k in names
            @assert length(unique(map(res->res[i][k][1], results_set))) == 1
            num_samples = results_set[1][i][k][1]

            error_set = map(res->abs(res[i][k][2]), results_set)
            μ_err, σ_err = mean_and_std(error_set)

            estimates_set = map(res->res[i][k][3], results_set)
            μ_est, σ_est = mean_and_std(estimates_set)

            num_failures_set = map(res->res[i][k][4], results_set)
            μ_num_failures, σ_num_failures = mean_and_std(num_failures_set)

            failure_rate_set = map(res->res[i][k][5], results_set)
            μ_failure_rate, σ_failure_rate = mean_and_std(failure_rate_set)

            ℓ_most_likely_failure_set = map(res->res[i][k][6], results_set)
            μ_ℓ_most_likely_failure, σ_ℓ_most_likely_failure = mean_and_std(ℓ_most_likely_failure_set)

            coverage_set = map(res->res[i][k][7], results_set)
            μ_coverage, σ_coverage = mean_and_std(coverage_set)

            region_set = map(res->res[i][k][8], results_set)
            μ_region, σ_region = mean_and_std(region_set)

            combined_results[i][k] = [num_samples, μ_err, σ_err, μ_est, σ_est, μ_num_failures, σ_num_failures, μ_failure_rate, σ_failure_rate, μ_ℓ_most_likely_failure, σ_ℓ_most_likely_failure, μ_coverage, σ_coverage, μ_region, σ_region]
        end
    end
    return combined_results
end


BASELINE_LABEL_STYLE = Dict(
    "lhc"=>"LHC",
    "sobol"=>"Sobol",
    "discrete"=>"Discrete",
    "uniform"=>"Uniform",
)


function plot_experiment_error(results; is_combined=false, show_ribbon=true, colors=nothing, apply_smoothing=false, use_stderr=true)
    plot()
    names = keys(first(results)) # same keys throughout
    for (i,k) in enumerate(names)
        # plot by key first
        X = []
        Y = []
        ribbon = []
        for r in eachindex(results)
            if is_combined
                x, μ, σ, _, _ = results[r][k]
                push!(Y, abs(μ))
                if show_ribbon
                    error_value = σ
                    if use_stderr
                        error_value = error_value/sqrt(x)
                    end
                    if μ - error_value <= 0
                        # Fix log-scale ribbon issues
                        error_value = μ - 1.1e-5
                    end
                    push!(ribbon, error_value)
                end
            else
                x, y = results[r][k]
                push!(Y, abs(y))
            end
            push!(X, x)
        end

        if apply_smoothing
            # First plot thin, unsmoothed values
            plot!(X, Y, lw=1, alpha=0.3, c=isnothing(colors) ? i : colors[i], label=false)
        end

        Y = apply_smoothing ? smooth(Y) : Y
        ribbon = apply_smoothing ? smooth(ribbon) : ribbon
        plot!(X, Y, ribbon=isempty(ribbon) ? nothing : ribbon, fillalpha=0.1, label=BASELINE_LABEL_STYLE[k], lw=2, c=isnothing(colors) ? i : colors[i])
        plot!(X, Y .+ ribbon; lw=1, alpha=0.5, c=colors[i], label=false)
        plot!(X, Y .- ribbon; lw=1, alpha=0.5, c=colors[i], label=false)
    end
    return plot!(xlabel="number of samples", ylabel="|error|", yaxis=:log, size=(600,300))
end


function plot_experiment_estimates(results, sparams, models; is_combined=false, show_ribbon=true)
    plot()
    names = keys(first(results)) # same keys throughout
    for k in names
        # plot by key first
        X = []
        Y = []
        ribbon = []
        for r in eachindex(results)
            if is_combined
                x, _, _, μ, σ = results[r][k]
                push!(Y, μ)
                if show_ribbon
                    push!(ribbon, σ/sqrt(x)) # standard error
                end
            else
                x, _, y = results[r][k]
                push!(Y, y)
            end
            push!(X, x)
        end
        Y = replace(Y, 0.0=>1e-5)
        plot!(X, Y, ribbon=isempty(ribbon) ? nothing : ribbon, label=k, lw=3)
    end
    truth = truth_estimate(sparams, models)
    hline!([truth], c=:black, lw=2, ls=:dash, alpha=0.8, label="truth")
    return plot!(xlabel="number of samples", ylabel="p(fail) estimate", legend=:topright, yaxis=:log, size=(600,300))
end


function recompute_p_estimates(gp, models; step=3)
    iterations = []
    p_estimates = []
    for i in 3:step:length(gp.y)
        @info "Iteration $i"
        X = gp.x[:, 1:i]
        Y = inverse.(gp.y[1:i])
        gp′ = gp_fit(X, Y)
        p_fail = p_estimate(gp′, models)
        @info "p(fail) = $p_fail"
        push!(iterations, i)
        push!(p_estimates, p_fail)
    end

    return iterations, p_estimates
end


function plot_p_estimates(iterations, p_estimates)
    plot(iterations, p_estimates,
        c=:darkgreen,
        label=false,
        lw=3,
        margin=5Plots.mm,
        size=(700,300),
        xlabel="number of samples",
        ylabel="\$\\hat{P}_{fail}\$",
    )
end

BASELINE_COLS = distinguishable_colors(12, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
BASELINE_PCOLS = map(col -> RGB(red(col), green(col), blue(col)), BASELINE_COLS)

function plot_combined_ablation_and_baseline_results(ar, cr; apply_smoothing=false)
    colors = cgrad(:Set1_5, 5, categorical=true)[[1,2,4,5]] # [:magenta, :blue, :red, :cyan]
    plot_experiment_error(cr; is_combined=true, colors=colors, apply_smoothing)
    plot_ablation(ar; acqs_set=[[1,2,3]], hold=true, label="FSAR", c=:darkgreen, apply_smoothing)
    plot!(
        xlabel="number of samples",
        ylabel="absolute error",
        size=(700,300),
        yaxis=:log,
        margin=5Plots.mm
    )
end


function plot_combined_ablation_and_baseline_estimate(ar, cr)
    colors = cgrad(:Set1_5, 5, categorical=true)[[1,2,4,5]] # [:magenta, :blue, :red, :cyan]
    plot_experiment_error(cr; is_combined=true, colors=colors)
    plot_ablation(ar; acqs_set=[[1,2,3]], hold=true, label="FSAR", c=:darkgreen)
    plot!(
        xlabel="number of samples",
        ylabel="absolute error",
        size=(700,300),
        yaxis=:log,
        margin=5Plots.mm
    )
end


function experiments_latex_table(cr)
    table = ""
    rd = x->round(x; sigdigits=3)
    rd4 = x->round(x; sigdigits=4)
    names = keys(first(cr))
    for k in names
        data = cr[end][k] # last iteration
        num_samples, μ_err, σ_err, μ_est, σ_est, μ_nfail, σ_nfail, μ_rfail, σ_rfail, μ_mlfl, σ_mlfl, μ_coverage, σ_coverage, μ_region, σ_region = data
        # TODO: arrows
        table *= "\$$(k)\$  &  \$$(rd(μ_rfail))\$  &  \$$(rd(μ_mlfl))\$  &  \$$(rd(μ_err))\$  &  \$$(rd(μ_coverage))\$  &  \$$(rd4(μ_region))\$  \\\\\n"
    end
    return table
end
