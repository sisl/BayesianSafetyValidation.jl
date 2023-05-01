module BayesianFailureProbability

using Reexport
using Alert
using ColorSchemes
@reexport using Distributions
using GaussianProcesses
using KernelDensity
using LatinHypercubeSampling
using LinearAlgebra
using Optim
@reexport using Parameters
using Plots
using Random
using Sobol
using StatsBase
using Suppressor

include("parameters.jl")
include("systems/system.jl")
include("surrogate_model.jl")
include("likelihood_weighting.jl")
include("importance_sampling.jl")
include("utils.jl")
include("metrics.jl")
include("plotting.jl")
include("experiments/experiments.jl")
include("experiments/baselines.jl")
include("experiments/ablations.jl")
include("experiments/pmc.jl")
include("iteratively_sample.jl")

using .System

export
    ## system.jl
    System,

    ## parameters.jl
    OperationalParameters,

    ## surrogate_model.jl
    logit,
    inverse_logit,
    transform,
    inverse_transform,
    apply,
    inverse,
    gp_fit,
    predict_f_vec,
    f_gp,
    σ²_gp,
    g_gp,
    gp_output,
    p_output,
    ucb,
    lcb,
    uncertainty_acquisition,
    boundary_acquisition,
    failure_region_sampling_acquisition,
    multi_objective_acqusition,
    get_next_point,
    sample_next_point,

    ## likelihood_weighting.jl
    p_estimate,
    lw_statistics,
    p_estimate_biased,
    p_estimate_samples,
    p_estimate_naive,

    ## importance_sampling.jl
    is_estimate_uniform,
    is_estimate_q,
    is_statistics,
    mc_estimte,
    q_proposal,
    is_estimate_theoretically_optimal,

    ## utils.jl
    create_gif,
    sigmoid,
    normalize01,

    ## metrics.jl
    falsification,
    most_likely_failure,
    most_likely_failures,
    most_likely_failure_likelihood,
    coverage,
    region_characterization,
    compute_metrics,

    ## plotting.jl
    get_model_ranges,
    plot_data!,
    plot_soft_boundary,
    plot_hard_boundary,
    plot_truth,
    plot_acquisition,
    plot_model,
    plot_combined,
    plot_surrogate_truth_combined,
    plot_q_proposal,
    plot_most_likely_failure,
    savefig_dense,

    ## experiments.jl
    run_experiment,
    run_experiments_rng,
    combine_results,
    plot_experiment_error,
    plot_experiment_estimates,
    run_pmc,
    run_mc,
    pmc,
    snis,
    gfunc,
    run_pmc_experiment,
    plot_estimate_curves,
    recompute_p_estimates,
    plot_p_estimates,
    plot_distribution_of_failures,
    plot_combined_ablation_and_baseline_results,
    experiments_latex_table,

    ## baselines.jl
    baseline_discrete,
    baseline_uniform,
    baseline_sobol,
    baseline_lhc,
    run_baseline,
    run_baselines,
    test_baselines,
    truth_estimate,
    plot_baselines,

    ## ablations.jl
    run_acquisition_ablations,
    plot_ablation,
    ablation_latex_table,

    ## iteratively_sample.jl
    iteratively_sample

end # module
