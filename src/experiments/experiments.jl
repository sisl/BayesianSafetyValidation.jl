function run_experiment(sparams, models; T=200, seed=0, record_every=10, show_plots=false, skip_gp=false, save_final_gp=true)
    Random.seed!(seed)
    results = []
    gp = nothing
    samples_to_run = 0
    while true
        if skip_gp
            samples_to_run = min(samples_to_run + 3record_every, 3T)
        else
            @time gp = bayesian_safety_validation(sparams, models;
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


function plot_experiment_error(results; is_combined=false, show_ribbon=true, colors=nothing, apply_smoothing=false, use_stderr=false, relative_error=true, truth=1)
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
                if relative_error
                    y = abs(μ)/truth
                    σ = σ/truth
                else
                    y = abs(μ)
                end
                push!(Y, y)
                if show_ribbon
                    error_value = σ
                    if use_stderr
                        error_value = error_value/sqrt(x)
                    end
                    if y - error_value <= 0
                        # Fix log-scale ribbon issues
                        error_value = y - 1.0e-3
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
        plot!(X, Y, ribbon=isempty(ribbon) ? nothing : ribbon, fillalpha=0.05, label=BASELINE_LABEL_STYLE[k], lw=2, c=isnothing(colors) ? i : colors[i])
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


function recompute_p_estimates(gp, models; weights=missing, step=3, num_steps=500, gp_args=missing, hard=true, verbose=false)
    num_samples = []
    p_estimates = []
    p_estimate_confs = []
    for i in 3:step:length(gp.y)
        verbose && @info "Sample $i/$(length(gp.y))"
        X = gp.x[:, 1:i]
        Z = gp.y[1:i]
        Y = inverse.(gp.y[1:i])
        if ismissing(weights)
            if ismissing(gp_args)
                gp′ = gp_fit(X, Y)
            else
                gp′ = gp_fit(X, Y; gp_args...)
            end
        else
            # SNIS does not need re-fit GP
            gp′ = (x=X, y=Z)
        end
        p_fail, p_fail_conf = p_estimate(gp′, models; num_steps, hard, weights=weights[1:i])
        verbose && @info "p(fail) = $p_fail"
        push!(num_samples, i)
        push!(p_estimates, p_fail)
        push!(p_estimate_confs, p_fail_conf)
    end

    return num_samples, p_estimates, p_estimate_confs
end


function plot_p_estimates(num_samples, p_estimates, p_estimates_conf; gpy=missing, nominal=missing, scale=1.5, full_nominal=false, logscale=true)
    show_nominal = !ismissing(nominal)

    if show_nominal
        nominal_color = :black
        nominal_end = mean(nominal)[end]

        # Actual nominal line hline
        plot([1, length(nominal)], [nominal_end, nominal_end]; label=false, c=nominal_color, lw=2, ls=:dash, alpha=0.5)

        if !full_nominal
            nominal = nominal[1:min(length(nominal), num_samples[end])]
        end
        plot_nominal(nominal)
        plotf = plot!
    else
        plotf = plot
    end

    bsv_color = :darkgreen
    bsv_ls = (c=bsv_color, lw=1)

    # Dummy without ribbon for legend
    # plt = plotf([-100, -100], [0, 0]; bsv_ls..., label=show_nominal ? "BSV estimate" : false)
    # plot!(num_samples, p_estimates;
    plt = plotf(num_samples, p_estimates;
        bsv_ls...,
        label=show_nominal ? "BSV estimate" : false,
        # margin=5Plots.mm,
        # size=(700,300), # was (700,300) for RWD plot
        size=(600,350) ./ scale,
        xlabel="number of samples",
        ylabel="p(fail) estimate",
        ribbon=p_estimates_conf,
        fillalpha=0.2,
        margin=scale ≥ 1.5 ? 1Plots.mm : 15Plots.mm,
        legend=:topright,
    )

    if show_nominal
        xlims!(1, length(nominal))
    else
        xlims!(1, num_samples[end])
    end

    xl = xlims()

    if !ismissing(gpy)
        cs0 = (y, cs=Float64.(cumsum(y)), ϵ=1e-1) -> begin
            cs[cs .== 0] .= ϵ
            return cs
        end

        twinls = :dot
        plttwindata = (eachindex(gpy), cs0(inverse.(gpy)))
        if logscale
            plttwinargs = (xlims=xl, yaxis=:log, label=false)
        else
            plttwinargs = (xlims=xl, label=false)
        end
        plttwin = plot(plttwindata...; plttwinargs..., bsv_ls..., alpha=0.5)
        yl = ylims()
        if logscale
            yl = (yl[1], yl[2]*100_000_000_000_000)
        else
            yl = (yl[1], yl[2]*3)
        end
        ylims!(yl)
        plt = plot(plt)
        plot!(twinx(), plttwindata...; plttwinargs..., ylims=yl, ylabel="number of failures", bsv_ls..., ls=twinls)
        if show_nominal
            plot!(twinx(), eachindex(nominal), cs0(nominal); xlims=xl, ylims=yl, yaxis=:log, ticks=false, label=false, c=nominal_color, lw=1, ls=twinls, alpha=1.0)
        end
    end

    return plt
end


function plot_nominal(nominal; hold=true)
    nominal_μs = [mean(nominal[1:i]) for i in eachindex(nominal)];
    nominal_conf = [i == 1 ? 0 : 2.58 * (std(nominal[1:i]) / sqrt(i)) for i in eachindex(nominal)];

    plotf = hold ? plot! : plot
    # Dummy without ribbon for legend
    # plotf([-100, -100], [0, 0], c=:black, ls=:dash, label="nominal estimate")
    # plot!(eachindex(nominal), nominal_μs, ribbon=nominal_conf, c=:black, label=false, ls=:dash, lw=1, fillalpha=0.2)
    plotf(eachindex(nominal), nominal_μs, ribbon=nominal_conf, c=:black, label="nominal estimate", ls=:solid, lw=1, fillalpha=0.2)
end


BASELINE_COLS = distinguishable_colors(12, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
BASELINE_PCOLS = map(col -> RGB(red(col), green(col), blue(col)), BASELINE_COLS)

function plot_combined_ablation_and_baseline_results(ar, cr, sparams=nothing, models=nothing; apply_smoothing=true, relative_error=true)
    colors = cgrad(:Set1_5, 5, categorical=true)[[1,2,4,5]] # [:magenta, :blue, :red, :cyan]
    truth = relative_error ? truth_estimate(sparams, models) : 0
    plot_experiment_error(cr; is_combined=true, colors=colors, apply_smoothing, relative_error, truth)
    plot_ablation(ar; acqs_set=[[1,2,3]], hold=true, label="FSAR", c=:darkgreen, apply_smoothing, relative_error, truth)
    plot!(
        xlabel="number of samples",
        ylabel=relative_error ? "relative error" : "absolute error",
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


function experiments_latex_table(cr, sparams=nothing, models=nothing; relative_error=true)
    if relative_error
        truth = truth_estimate(sparams, models)
    else
        truth = 1 # divisor
    end
    table = ""
    rd = x->round(x; sigdigits=3)
    rd4 = x->round(x; sigdigits=4)
    rd5 = x->round(x; sigdigits=5)
    names = keys(first(cr))
    for k in names
        data = cr[end][k] # last iteration
        num_samples, μ_err, σ_err, μ_est, σ_est, μ_nfail, σ_nfail, μ_rfail, σ_rfail, μ_mlfl, σ_mlfl, μ_coverage, σ_coverage, μ_region, σ_region = data
        if relative_error
            μ_err = μ_err / truth
            σ_err = σ_err / truth
        end
        table *= "\$$(k)\$  &  \$$(rd(μ_rfail))\$  &  \$$(rd(μ_mlfl))\$  &  \$$(rd5(μ_err))\$  &  \$$(rd(μ_coverage))\$  &  \$$(rd4(μ_region))\$  \\\\\n"
    end
    return table
end
