using Revise
using BayesianFailureProbability

include("systems/dummy_booth_system.jl")
# include("systems/dummy_squares_system.jl")
# include("systems/dummy_himmelblau_system.jl")
# include("systems/dummy_himmelblau_probability_system.jl")
# include("systems/dummy_linear_system.jl")

@time gp = iteratively_sample(system_params, models; T=30, show_tight_combined_plot=true)

baselines, errors = run_baselines(gp, system_params, models, gp.nobs);
