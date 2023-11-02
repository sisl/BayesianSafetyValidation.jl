using Revise
using BayesianSafetyValidation
import AbstractGPs.KernelFunctions: ColVecs

Base.vcat(x1::ColVecs, x2::ColVecs) = ColVecs(hcat(x1.X, x2.X))

include("../src/systems/dummy_booth_system.jl")
# include("../src/systems/dummy_squares_system.jl")
# include("../src/systems/dummy_himmelblau_system.jl")
# include("../src/systems/dummy_himmelblau_probability_system.jl")
# include("../src/systems/dummy_linear_system.jl")

@time gp = bayesian_safety_validation(system_params, models; T=30, show_tight_combined_plot=true)

# baselines, errors = run_baselines(gp, system_params, models, gp.nobs)
