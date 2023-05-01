using Revise
using BayesianSafetyValidation
using BSON

include(joinpath(@__DIR__, "../src/systems/dummy_squares_system.jl"))
ar = run_acquisition_ablations(system_params, models, 5; T=30, record_every=30)
BSON.@save "ablation_single.bson" ar

ablation_latex_table(ar) |> println
