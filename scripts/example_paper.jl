using Revise
using BayesianSafetyValidation

@with_kw mutable struct DummyPaperSystem <: System.SystemParameters
    x1c = 2
    x2c = 5
end

System.generate_input(sparams::DummyPaperSystem, sample::Vector; kwargs...) = sample # pass-through
function System.reset(::DummyPaperSystem) end
function System.initialize(; kwargs...) end
function System.evaluate(sparams::DummyPaperSystem, inputs::Vector; kwargs...)
    return [x[1] ≥ sparams.x1c && x[2] ≥ sparams.x2c for x in inputs]
end

system_params = DummyPaperSystem()
px1 = OperationalParameters("distance", [0.1, 4], TruncatedNormal(0, 1.0, 0, 4))
px2 = OperationalParameters("slope", [1, 7], Normal(3, 0.5))
model = [px1, px2]

surrogate  = bayesian_safety_validation(system_params, model; T=30)
X_failures = falsification(surrogate.x, surrogate.y)
ml_failure = most_likely_failure(surrogate.x, surrogate.y, model)
p_failure  = p_estimate(surrogate, model)

truth = truth_estimate(system_params, model) # when using simple systems
plot_surrogate_truth_combined(surrogate, model, system_params; hide_model=false)
