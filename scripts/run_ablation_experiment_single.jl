using Revise
using BayesianSafetyValidation
using BSON

include(joinpath(@__DIR__, "../src/systems/dummy_squares_system.jl"))

RERUN = false

if RERUN
    ar = run_acquisition_ablations(system_params, models, 5; T=30, record_every=30)
    BSON.@save "ablation_single.bson" ar
else
    BSON.@load joinpath(@__DIR__, "../data/ablation_single.bson") ar
end

ablation_latex_table(ar, system_params, models) |> println
