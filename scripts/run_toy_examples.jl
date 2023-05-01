using Revise
using BSON
using BayesianSafetyValidation

config = (T=333, show_tight_combined_plot=false, save_plots_svg=false, hide_model_after_first=false)

include("../src/systems/dummy_booth_system.jl")
@time gp = bayesian_safety_validation(system_params, models; config...)
BSON.@save "gp_booth_T333.bson" gp

include("../src/systems/dummy_squares_system.jl")
@time gp = bayesian_safety_validation(system_params, models; config...)
BSON.@save "gp_squares_T333.bson" gp

include("../src/systems/dummy_himmelblau_system.jl")
@time gp = bayesian_safety_validation(system_params, models; config...)
BSON.@save "gp_himmelblau_T333.bson" gp
