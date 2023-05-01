using Revise
using BayesianFailureProbability
using BSON

include(joinpath(@__DIR__, "../src/systems/dummy_booth_system.jl"))
ar_booth = run_acquisition_ablations(system_params, models, 3; T=333, record_every=10, acqs_set=[[1,2,3]])
BSON.@save "ablation_booth_T333_rng3_every10_final.bson" ar_booth

include(joinpath(@__DIR__, "../src/systems/dummy_squares_system.jl"))
ar_squares = run_acquisition_ablations(system_params, models, 3; T=333, record_every=10, acqs_set=[[1,2,3]])
BSON.@save "ablation_squares_T333_rng3_every10_final.bson" ar_squares

include(joinpath(@__DIR__, "../src/systems/dummy_himmelblau_system.jl"))
ar_himmelblau = run_acquisition_ablations(system_params, models, 3; T=333, record_every=10, acqs_set=[[1,2,3]])
BSON.@save "ablation_himmelblau_T333_rng3_every10_final.bson" ar_himmelblau
