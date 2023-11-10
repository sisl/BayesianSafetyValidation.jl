using Revise
using BSON
using BayesianSafetyValidation

include("../src/systems/dummy_shape_system.jl")

# reset_colors!()
# set_colors!(["#007662", :white, "#8c1515"])
set_colors!([:gray, :white, "#8c1515"])
# set_colors!([:black, :white, "#8c1515"])

gp_args = (σ=20, ℓ=5)
args = (sample_from_acquisitions=[false,false,false], m=120, sample_temperature=1, gp_args, use_optim=true, verbose=false, optim_options=Optim.Options(iterations=3000), acquisitions_to_run=[2])
N = 300
T = N÷length(args.acquisitions_to_run)
@time gp = bayesian_safety_validation(system_params, models; T, show_tight_combined_plot=true, plot_every=Inf, args...)
# BSON.@load joinpath(@__DIR__, "..", "gp_shape.bson") gp

ms = 0.5
run_baselines(gp, system_params, models, length(gp.y); copmute_truth=false, show_truth=true, show_plots=true, show_data=true, data_ms=ms, use_circles=true, soft=false, tight=true, gp_args)

plot_hard_boundary(gp, models; show_data=true, ms, tight=true) |> display


nothing # REPL
