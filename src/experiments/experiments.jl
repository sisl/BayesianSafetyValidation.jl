function run_experiment(sparams, models; M=200, seed=0, record_every=10)
    Random.seed!(seed)
    results = []
    gp = nothing
    while true
        @time gp = iteratively_sample(system_params, models;
            seed=seed,
            gp=gp, # continue from last GP
            M=record_every,
            skip_if_no_failures=false, # Just f̂*p
            # alternate_acquisitions=false, # Just f̂*p
            alternate_acquisitions=true, # All three acquisitions
            show_alert=false,
            show_plots=false,
            show_acquisition=false)

        exp_baselines, exp_errors, exp_estimates = run_baselines(gp, sparams, models, gp.nobs; is=false)

        # record the [number of data points, error, p(fail) estimate]
        result = merge(vcat, Dict(k=>gp.nobs for (k,gp) in exp_baselines), exp_errors, exp_estimates)
        result["GP"] = [gp.nobs, exp_errors["GP"], exp_estimates["GP"]]
        push!(results, result)

        if gp.nobs ≥ M
            @info "Finished experiment."
            break
        end
    end
    return results
end


function run_experiments_rng(sparams, models, num_seeds=3; M=200, record_every=10)
    results_set = []
    for seed in 1:num_seeds
        results = run_experiment(sparams, models; seed, M, record_every)
        push!(results_set, results)
    end
    cr = combine_results(results_set)
    display(plot_experiment_estimates(cr, sparams, models; is_combined=true))
    display(plot_experiment_error(cr; is_combined=true))
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
            combined_results[i][k] = [num_samples, μ_err, σ_err, μ_est, σ_est]
        end
    end
    return combined_results
end


function plot_experiment_error(results; is_combined=false, num_std=2, show_ribbon=true)
    plot()
    names = keys(first(results)) # same keys throughout
    for k in names
        # plot by key first
        X = []
        Y = []
        ribbon = []
        for r in eachindex(results)
            if is_combined
                x, μ, σ, _, _ = results[r][k]
                push!(Y, abs(μ))
                if show_ribbon
                    push!(ribbon, abs(num_std*σ)/sqrt(x)) # standard error
                end
            else
                x, y = results[r][k]
                push!(Y, abs(y))
            end
            push!(X, x)
        end
        plot!(X, Y, ribbon=isempty(ribbon) ? nothing : ribbon, label=k, lw=2)
    end
    return plot!(xlabel="number of samples", ylabel="|error|", yaxis=:log, size=(600,300))
end


function plot_experiment_estimates(results, sparams, models; is_combined=false, num_std=2, show_ribbon=true)
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
                    push!(ribbon, num_std*σ/sqrt(x)) # standard error
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
