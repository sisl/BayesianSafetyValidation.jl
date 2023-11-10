using Revise
using BayesianSafetyValidation

include("../src/systems/dummy_booth_system.jl")
# include("../src/systems/dummy_squares_system.jl")
# include("../src/systems/dummy_himmelblau_system.jl")
# include("../src/systems/dummy_himmelblau_probability_system.jl")
# include("../src/systems/dummy_linear_system.jl")

@time gp = bayesian_safety_validation(system_params, models; T=30, show_tight_combined_plot=true)
