using BayesianFailureProbability

# include("systems/dummy_squares_system.jl")
# include("systems/dummy_booth_system.jl")
include("systems/dummy_linear_system.jl")

@time gp = iteratively_sample(system_params, models; M=33, show_combined_plot=true)

baselines, errors = run_baselines(gp, system_params, models, gp.nobs);
