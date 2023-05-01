function run_acquisition_ablation(sparams, models; T=200, seed=0, record_every=10, show_truth=true, acqs_set=[[1], [2], [3], [1,2], [2,3], [1,3], [1,2,3]])
    Random.seed!(seed)
    @info "Seed = $seed"
    results = Dict()
    gp = nothing

    if show_truth
        truth = truth_estimate(sparams, models)
        @info "truth est.: $truth"
    else
        truth = 0
    end

    for acqs in acqs_set
        gp = nothing
        results[acqs] = []
        while true
            T_intermediate = isnothing(gp) ? min(record_every, 3T÷length(acqs)) : min(record_every, (3T - length(gp.y))÷length(acqs))
            @time gp = bayesian_safety_validation(sparams, models;
                seed=seed,
                gp=gp, # continue from last GP
                T=T_intermediate,
                acquisitions_to_run=acqs)

            est = p_estimate(gp, models)
            gp_error = est - truth
            @info "GP ($acqs): $gp_error"

            num_failures = sum(gp.y .>= 0) # logits
            failure_rate = num_failures / length(gp.y)
            @info "GP ($acqs): Falsification = $num_failures ($failure_rate)"

            ℓ_most_likely_failure = most_likely_failure_likelihood(gp, models)
            @info "GP ($acqs): MLFA = $ℓ_most_likely_failure"

            coverage_metric = coverage(gp, models)
            @info "GP ($acqs): Coverage = $coverage_metric"

            region_metric = region_characterization(gp, models, sparams)
            @info "GP ($acqs): Region characterization = $region_metric"

            result = (gp.nobs, gp_error, est, num_failures, failure_rate, ℓ_most_likely_failure, coverage_metric, region_metric)
            push!(results[acqs], result)

            # if gp.nobs ≥ T
            if gp.nobs ≥ 3T # Make sure we run each experiment for the same number of observations (1 per acquisition, per `t`)
                @info "Finished experiment."
                break
            end
        end
    end
    return results
end


function run_acquisition_ablations(sparams, models, num_seeds=3; acqs_set=[[1], [2], [3], [1,2], [2,3], [1,3], [1,2,3]], kwargs...)
    results_set = []
    for seed in 0:num_seeds-1
        if seed > 0
            # Remove the acquisition functions that are deterministic
            for aq in [[1], [2], [1,2]]
                idx = findfirst(a->a == aq, acqs_set)
                if !isnothing(idx)
                    @info "Removing deterministic acqusition $aq for the remainder of the seeds."
                    deleteat!(acqs_set, idx)
                end
            end
        end
        results = run_acquisition_ablation(sparams, models; seed, acqs_set, kwargs...)
        push!(results_set, results)
    end
    ar = combine_ablation_results(results_set)
    alert("Ablations finished!")
    return ar
end



function combine_ablation_results(results_set::Vector)
    names = keys(results_set[1])
    combined_results = Dict()
    for k in names
        for i in eachindex(results_set)
            if haskey(results_set[i], k)
                combined_results[k] = Vector(undef, length(results_set[i][k]))
                for j in eachindex(results_set[i][k])

                    num_samples = results_set[i][k][j][1]

                    error_set = map(res->abs(res[k][j][2]), filter(res->haskey(res, k), results_set))
                    μ_err, σ_err = mean_and_std(error_set)

                    estimates_set = map(res->res[k][j][3], filter(res->haskey(res, k), results_set))
                    μ_est, σ_est = mean_and_std(estimates_set)

                    num_failures_set = map(res->res[k][j][4], filter(res->haskey(res, k), results_set))
                    μ_nfail, σ_nfail = mean_and_std(num_failures_set)

                    failure_rate_set = map(res->res[k][j][5], filter(res->haskey(res, k), results_set))
                    μ_rfail, σ_rfail = mean_and_std(failure_rate_set)

                    mlfl_set = map(res->res[k][j][6], filter(res->haskey(res, k), results_set))
                    μ_mlfl, σ_mlfl = mean_and_std(mlfl_set)

                    coverage_set = map(res->res[k][j][7], filter(res->haskey(res, k), results_set))
                    μ_coverage, σ_coverage = mean_and_std(coverage_set)

                    region_set = map(res->res[k][j][8], filter(res->haskey(res, k), results_set))
                    μ_region, σ_region = mean_and_std(region_set)

                    combined_results[k][j] = [num_samples, μ_err, σ_err, μ_est, σ_est, μ_nfail, σ_nfail, μ_rfail, σ_rfail, μ_mlfl, σ_mlfl, μ_coverage, σ_coverage, μ_region, σ_region]
                end
            end
        end
    end
    return combined_results
end


COLS = distinguishable_colors(7, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
PCOLS = map(col -> RGB(red(col), green(col), blue(col)), COLS)

ABLATION_STYLES = Dict(
    [1]=>(ls=:dot, c=PCOLS[1], label="ue"),
    [2]=>(ls=:dot, c=PCOLS[2], label="br"),
    [3]=>(ls=:dot, c=PCOLS[3], label="frs"),
    [1,2]=>(ls=:dash, c=PCOLS[4], label="ue, br"),
    [2,3]=>(ls=:dash, c=PCOLS[5], label="br, frs"),
    [1,3]=>(ls=:dash, c=PCOLS[6], label="ue, frs"),
    [1,2,3]=>(ls=:solid, c=:black, label="ue, br, frs"),
)

function plot_ablation(ar; hold=false, use_log=false, plot_pfail=false, apply_smoothing=false, title="", acqs_set=[[1], [2], [3], [1,2], [2,3], [1,3], [1,2,3]], label=nothing, c=nothing, show_error=true, use_stderr=true)
    hold ? plot!() : plot()
    yf = use_log ? log : y->y
    for k in acqs_set
        X = map(d->d[1], ar[k])
        Y = map(d->yf(d[plot_pfail ? 4 : 2]), ar[k])
        σerr = map(d->begin
            μ = yf(d[plot_pfail ? 4 : 2])
            σ = yf(d[plot_pfail ? 5 : 3])
            if use_stderr
                σ = σ / sqrt(d[1])
            end
            if μ - σ <= 0
                # Fix log-scale ribbon issues
                σ = μ - 1.1e-5
            end
            σ
        end, ar[k])

        style = deepcopy(ABLATION_STYLES[k])
        if !isnothing(label)
            # overwrite label
            style = (style..., label=label)
        end
        if !isnothing(c)
            # overwrite color
            style = (style..., c=c)
        end
        if apply_smoothing
            # First plot thin, unsmoothed values
            plot!(X, Y; lw=1, alpha=0.3, style..., label=false)
        end
        Y = apply_smoothing ? smooth(Y) : Y
        plot!(X, Y, ribbon=show_error ? σerr : nothing, fillalpha=0.1; lw=2, style...)
        if show_error
            plot!(X, Y .+ σerr; lw=1, alpha=0.5, style..., label=false)
            plot!(X, Y .- σerr; lw=1, alpha=0.5, style..., label=false)
        end
    end
    plot!(title=title)
end


function ablation_latex_table(ar; acqs_set=[[1], [2], [3], [1,2], [2,3], [1,3], [1,2,3]])
    table = ""
    rd = x->round(x; sigdigits=3)
    rd4 = x->round(x; sigdigits=4)

    for k in acqs_set
        data = ar[k][end] # last iteration
        num_samples, μ_err, σ_err, μ_est, σ_est, μ_nfail, σ_nfail, μ_rfail, σ_rfail, μ_mlfl, σ_mlfl, μ_coverage, σ_coverage, μ_region, σ_region = data
        tab = length(k) == 1 ? "\t\t" : "\t"
        table *= "\$$(k)\$  $tab&  \$$(rd(μ_rfail))\$  \t&  \$$(rd(μ_mlfl))\$  \t&  \$$(rd(μ_err))\$  \t&  \$$(rd(μ_coverage))\$  \t&  \$$(rd4(μ_region))\$  \\\\\n"
    end
    return table
end
