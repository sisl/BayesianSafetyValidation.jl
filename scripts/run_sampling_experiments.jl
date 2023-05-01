using Revise
using BayesianSafetyValidation
using BSON

include(joinpath(@__DIR__, "../src/systems/dummy_booth_system.jl"))
@time cr_booth = run_experiments_rng(system_params, models, 3; T=333, record_every=10, skip_gp=true)
BSON.@save "results_sampling_booth_T333_rng3_every10_final.bson" cr_booth

include(joinpath(@__DIR__, "../src/systems/dummy_squares_system.jl"))
@time cr_squares = run_experiments_rng(system_params, models, 3; T=333, record_every=10, skip_gp=true)
BSON.@save "results_sampling_squares_T333_rng3_every10_final.bson" cr_squares

include(joinpath(@__DIR__, "../src/systems/dummy_himmelblau_system.jl"))
@time cr_himmelblau = run_experiments_rng(system_params, models, 3; T=333, record_every=10, skip_gp=true)
BSON.@save "results_sampling_himmelblau_T333_rng3_every10_final.bson" cr_himmelblau
