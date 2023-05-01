using Revise
using BSON
using BayesianFailureProbability
using Plots; default(fontfamily="Computer Modern", framestyle=:box)

!isdir("truths") && mkdir("truths")

include("../src/systems/dummy_booth_system.jl")
BSON.@load "gp_booth_T333.bson" gp
plot_combined(gp, models, system_params; truth=true, tight=true, overlay=true, hide_ranges=true, latex_labels=true, include_surrogate=true, show_data=false)
savefig("truths/truth_booth_models_joint.svg")
plot_hard_boundary(gp, models; show_data=false, tight=true)
# plot_combined(gp, models, system_params; soft=false, tight=true, overlay=true, label1=false, label2=false, show_data=false, hide_ranges=true) # Note: If you want to include the op. models
savefig("truths/surrogate_hard_booth.svg")
plot_hard_boundary(gp, models; show_data=false, tight=true, lw=0.1)
savefig("truths/surrogate_hard_booth_small.svg")

include("../src/systems/dummy_squares_system.jl")
BSON.@load "gp_squares_T333.bson" gp
plot_combined(gp, models, system_params; truth=true, tight=true, overlay=true, hide_ranges=true, latex_labels=true, include_surrogate=true, show_data=false)
savefig("truths/truth_squares_models_joint.svg")
plot_hard_boundary(gp, models; show_data=false, tight=true)
savefig("truths/surrogate_hard_squares.svg")
# plot_combined(gp, models, system_params; soft=false, tight=true, overlay=true, label1=false, label2=false, show_data=false, hide_ranges=true) # Note: If you want to include the op. models
plot_hard_boundary(gp, models; show_data=false, tight=true, lw=0.1)
savefig("truths/surrogate_hard_squares_small.svg")

include("../src/systems/dummy_himmelblau_system.jl")
BSON.@load "gp_himmelblau_T333.bson" gp
plot_combined(gp, models, system_params; truth=true, tight=true, overlay=true, overlay_levels=200, hide_ranges=true, latex_labels=true, include_surrogate=true, show_data=false)
savefig("truths/truth_himmelblau_models.svg")
plot_hard_boundary(gp, models; show_data=false, tight=true)
# plot_combined(gp, models, system_params; soft=false, tight=true, overlay=true, label1=false, label2=false, show_data=false, hide_ranges=true) # Note: If you want to include the op. models
savefig("truths/surrogate_hard_himmelblau.svg")
plot_hard_boundary(gp, models; show_data=false, tight=true, lw=0.1)
savefig("truths/surrogate_hard_himmelblau_small.svg")
