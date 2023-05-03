using Revise
using BayesianSafetyValidation
using BSON
using Plots; default(fontfamily="Computer Modern", framestyle=:box)

use_latex = false

if use_latex
    include("pgfplots_config.jl")
else
    Plots.reset_defaults()
    using Plots; default(fontfamily="Computer Modern", framestyle=:box)
    gr()
end


@info "Booth"
include("../src/systems/dummy_booth_system.jl")
BSON.@load joinpath(@__DIR__, "../data/ablation_booth_T333_rng3_every10_final.bson") ar_booth
BSON.@load joinpath(@__DIR__, "../data/results_sampling_booth_T333_rng3_every10_final.bson") cr_booth
plot_combined_ablation_and_baseline_results(ar_booth, cr_booth, system_params, models)
title!("\\textsc{Representative}")
plot!(legend=:bottomleft)
if use_latex
    savefig("baseline_booth.tex")
else
    display(plot!())
end
experiments_latex_table(cr_booth, system_params, models) |> println
ablation_latex_table(ar_booth, system_params, models; acqs_set=[[1,2,3]]) |> println


@info "Squares"
include("../src/systems/dummy_squares_system.jl")
BSON.@load joinpath(@__DIR__, "../data/ablation_squares_T333_rng3_every10_final.bson") ar_squares
BSON.@load joinpath(@__DIR__, "../data/results_sampling_squares_T333_rng3_every10_final.bson") cr_squares
plot_combined_ablation_and_baseline_results(ar_squares, cr_squares, system_params, models)
title!("\\textsc{Squares}")
plot!(legend=:bottomleft)
if use_latex
    savefig("baseline_squares.tex")
else
    display(plot!())
end
experiments_latex_table(cr_squares, system_params, models) |> println
ablation_latex_table(ar_squares, system_params, models; acqs_set=[[1,2,3]]) |> println


@info "Himmelblau"
include("../src/systems/dummy_himmelblau_system.jl")
BSON.@load joinpath(@__DIR__, "../data/ablation_himmelblau_T333_rng3_every10_final.bson") ar_himmelblau
BSON.@load joinpath(@__DIR__, "../data/results_sampling_himmelblau_T333_rng3_every10_final.bson") cr_himmelblau
plot_combined_ablation_and_baseline_results(ar_himmelblau, cr_himmelblau, system_params, models)
title!("\\textsc{Mixture}")
plot!(legend=:bottomleft)
if use_latex
    savefig("baseline_himmelblau.tex")
else
    display(plot!())
end
experiments_latex_table(cr_himmelblau, system_params, models) |> println
ablation_latex_table(ar_himmelblau, system_params, models; acqs_set=[[1,2,3]]) |> println
