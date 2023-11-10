include("dummy_himmelblau_system.jl")

@with_kw mutable struct DummyHimmelblauProbabilitySystem <: System.SystemParameters
    γ = DummyHimmelblauSystem().γ
end

system_params = DummyHimmelblauProbabilitySystem()
θ1 = OperationalParameters("x_1", [-6, 6], Uniform(-6, 6))
θ2 = OperationalParameters("x_2", [-6, 6], Uniform(-6, 6))
models = [θ1, θ2]

f_himmelblau_prob(sparams::DummyHimmelblauProbabilitySystem, x; fx=himmelblau(x)) = inverse_logit(sparams.γ - fx; s=1/sparams.γ)

function System.evaluate(sparams::DummyHimmelblauProbabilitySystem, inputs::Vector; verbose=false, kwargs...)
    verbose && @info "Evaluating dummy Himmelblau probability system ($inputs)..."
    Y = Vector{Real}(undef, 0)
    for input in inputs
        failure = f_himmelblau_prob(sparams, input)
        push!(Y, failure)
    end
    return Y
end
