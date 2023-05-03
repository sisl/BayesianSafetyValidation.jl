function pmc(p; N::Int=8, T::Int=10, σ::Real=1.0, dₓ::Int=2, use_dm=false)
	μ = [rand(p) for _ in 1:N]
	Σ = σ^2*I(dₓ)
	S = []
	for t in 1:T
		qₜ = [MvNormal(μ[n], Σ) for n in 1:N]
		xₜ = [rand(qₜ[n]) for n in 1:N]
		if use_dm # Deterministic mixture multiple importance sampling (DM-MIS)
			wₜ = [pdf(p, xₜ[n]) / mean(pdf(qₜ[j], xₜ[n]) for j in 1:N) for n in 1:N]
		else # Standard multimple importance sampling (s-MIS)
			wₜ = [pdf(p, xₜ[n]) / pdf(qₜ[n], xₜ[n]) for n in 1:N]
		end
		push!(S, (xₜ, wₜ))
		μ = sample(xₜ, Weights(wₜ), N, replace=true)
	end
	samples = reduce(vcat, first.(S))
	weights = reduce(vcat, last.(S))
	return samples, weights
end

snis(g, X, W) = sum(g(X[k]) * W[k] for k in eachindex(X)) / sum(W)
stderr(X) = std(X) / sqrt(length(X))

PMC_DEFAULTS = (T_max=100, N_seeds=3, N_qs=50)

function run_pmc(sparams, p; T_max=PMC_DEFAULTS.T_max, N_seeds=PMC_DEFAULTS.N_seeds, N_qs=PMC_DEFAULTS.N_qs)
    g = gfunc(sparams)
    μ_iters = []
    stderr_iters = []
    num_evals = 0
    X_samples = []
    for t in 1:T_max
        estimates = []
        local X
        for i in 1:N_seeds
            Random.seed!(i)
            X, W = pmc(p; N=N_qs, T=t, σ=5.0)
            μ̃ = snis(g, X, W)
            push!(estimates, μ̃)
        end
        num_evals += length(X) # Only once per seed (don't double count!)
        push!(X_samples, num_evals)
        estimate = mean(estimates)
        estimate_stderr = stderr(estimates)
        push!(μ_iters, estimate)
        push!(stderr_iters, estimate_stderr)
    end
    return X_samples, μ_iters, stderr_iters, num_evals
end


function run_mc(sparams, p; T_max=PMC_DEFAULTS.T_max, N_seeds=PMC_DEFAULTS.N_seeds, N_qs=PMC_DEFAULTS.N_qs)
    g = gfunc(sparams)
    μ_iters_mc = []
	stderr_iters_mc = []
	num_evals_mc = 0
	X_samples_mc = []
	for t in 1:T_max
		estimates = []
        local M_mc
		for i in 1:N_seeds
			Random.seed!(i)
			M_mc = t*N_qs # NOTE: (t*N_qs) to match PMC number of evals.
            μ̃ = mean(g(xy) for xy in rand(p, M_mc))
			push!(estimates, μ̃)
		end
        num_evals_mc += M_mc # Only once per seed (don't double count!)
		push!(X_samples_mc, num_evals_mc)
		estimate = mean(estimates)
		estimate_stderr = stderr(estimates)
		push!(μ_iters_mc, estimate)
		push!(stderr_iters_mc, estimate_stderr)
	end
	X_samples_mc, μ_iters_mc, stderr_iters_mc, num_evals_mc
end


gfunc(sparams::System.SystemParameters) = x -> System.evaluate(sparams, System.generate_input(sparams, [x]), verbose=false)


function run_pmc_experiment(sparams, models; kwargs...)
    res_pmc = run_pmc(sparams, models; kwargs...)
    res_mc = run_mc(sparams, models; kwargs...)
    return res_pmc, res_mc
end


function plot_estimate_curves(ar, res_pmc, res_mc, system_params, models; relative_error=true, use_stderr=true)
    truth = truth_estimate(system_params, models)

    X_mc = deepcopy(res_mc[1])
    Y_mc = abs.(truth .- first.(res_mc[2]))
    Y_mc_err = first.(res_mc[3])
    if !use_stderr
        Y_mc_err = Y_mc_err .* sqrt.(X_mc)
    end

    X_pmc = deepcopy(res_pmc[1])
    Y_pmc = abs.(truth .- first.(res_pmc[2]))
    Y_pmc_err = first.(res_pmc[3])
    if !use_stderr
        Y_pmc_err = Y_pmc_err .* sqrt.(X_pmc)
    end

    X_bsv = map(r->r[1], ar[[1,2,3]])
    Y_bsv = map(r->r[2], ar[[1,2,3]])
    Y_bsv_err = map(r->r[3], ar[[1,2,3]])
    if use_stderr
        Y_bsv_err = Y_bsv_err ./ sqrt.(X_bsv) # convert to stderr
    end

    if relative_error
        Y_mc = Y_mc ./ truth
        Y_mc_err = Y_mc_err ./ truth

        Y_pmc = Y_pmc ./ truth
        Y_pmc_err = Y_pmc_err ./ truth

        Y_bsv = Y_bsv ./ truth
        Y_bsv_err = Y_bsv_err ./ truth
    end

    plot()

    plot!(X_mc, Y_mc, ribbon=Y_mc_err, fillalpha=0.2, lw=2, label="MC", c=:gray)
    plot!(X_mc, Y_mc + Y_mc_err, lw=1, alpha=0.5, label=false, c=:gray)
    plot!(X_mc, Y_mc - Y_mc_err, lw=1, alpha=0.5, label=false, c=:gray)

    plot!(X_pmc, Y_pmc, ribbon=Y_pmc_err, fillalpha=0.2, lw=2, label="PMC", c=:crimson)
    plot!(X_pmc, Y_pmc + Y_pmc_err, lw=1, alpha=0.5, label=false, c=:crimson)
    plot!(X_pmc, Y_pmc - Y_pmc_err, lw=1, alpha=0.5, label=false, c=:crimson)

    plot!(X_bsv, Y_bsv, ribbon=Y_bsv_err, fillalpha=0.2, lw=2, label="BSV", c=:darkgreen)
    plot!(X_bsv, Y_bsv + Y_bsv_err, lw=1, alpha=0.5, label=false, c=:darkgreen)
    plot!(X_bsv, Y_bsv - Y_bsv_err, lw=1, alpha=0.5, label=false, c=:darkgreen)

    plot!(xlabel="number of samples", ylabel=relative_error ? "relative error" : "absolute error")
    plot!(xticks=[1, 10, 100, 1000, 10_000, 100_000, 1_000_000], xaxis=:log, size=(700,300), margin=3Plots.mm)
end
