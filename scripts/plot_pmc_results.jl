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
res_pmc, res_mc = run_pmc_experiment(system_params, models)
plot_estimate_curves(ar_booth, res_pmc, res_mc, system_params, models)
title!("\\textsc{Representative}")
if use_latex
    savefig("pmc_booth.tex")
else
    display(plot!())
end

@info "Squares"
include("../src/systems/dummy_squares_system.jl")
BSON.@load joinpath(@__DIR__, "../data/ablation_squares_T333_rng3_every10_final.bson") ar_squares
res_pmc, res_mc = run_pmc_experiment(system_params, models)
plot_estimate_curves(ar_squares, res_pmc, res_mc, system_params, models)
title!("\\textsc{Squares}")
if use_latex
    savefig("pmc_squares.tex")
else
    display(plot!())
end

@info "Himmelblau"
include("../src/systems/dummy_himmelblau_system.jl")
BSON.@load joinpath(@__DIR__, "../data/ablation_himmelblau_T333_rng3_every10_final.bson") ar_himmelblau
res_pmc, res_mc = run_pmc_experiment(system_params, models)
plot_estimate_curves(ar_himmelblau, res_pmc, res_mc, system_params, models)
title!("\\textsc{Mixture}")
if use_latex
    savefig("pmc_himmelblau.tex")
else
    display(plot!())
end

