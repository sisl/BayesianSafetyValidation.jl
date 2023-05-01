using Revise
using BayesianFailureProbability
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


@info "Himmelblau"
include("../src/systems/dummy_himmelblau_system.jl")
BSON.@load joinpath(@__DIR__, "ablation_himmelblau_T333_rng3_every10_final.bson") ar_himmelblau
BSON.@load "results_sampling_himmelblau_T333_rng3_every10_final.bson" cr_himmelblau
plot_combined_ablation_and_baseline_results(ar_himmelblau, cr_himmelblau)
title!("\\textsc{Mixture}")
plot!(legend=:bottomleft)
if use_latex
    savefig("baseline_himmelblau.tex")
else
    display(plot!())
end


@info "Booth"
include("../src/systems/dummy_booth_system.jl")
BSON.@load joinpath(@__DIR__, "ablation_booth_T333_rng3_every10_final.bson") ar_booth
BSON.@load "results_sampling_booth_T333_rng3_every10_final.bson" cr_booth
plot_combined_ablation_and_baseline_results(ar_booth, cr_booth)
title!("\\textsc{Representative}")
if use_latex
    savefig("baseline_booth.tex")
else
    display(plot!())
end

@info "Squares"
include("../src/systems/dummy_squares_system.jl")
BSON.@load joinpath(@__DIR__, "ablation_squares_T333_rng3_every10_final.bson") ar_squares
BSON.@load "results_sampling_squares_T333_rng3_every10_final.bson" cr_squares
plot_combined_ablation_and_baseline_results(ar_squares, cr_squares)
title!("\\textsc{Squares}")
if use_latex
    savefig("baseline_squares.tex")
else
    display(plot!())
end
