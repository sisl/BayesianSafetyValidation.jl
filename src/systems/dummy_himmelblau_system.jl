@with_kw mutable struct DummyHimmelblauSystem <: System.SystemParameters
    γ = 15
end

system_params = DummyHimmelblauSystem()
θ1 = OperationalParameters("x_1", [-6, 6], MixtureModel([TruncatedNormal(2.0, 1.0, -6, 6), TruncatedNormal(-2.0, 1.0, -6, 6)], [0.5, 0.5]))
θ2 = OperationalParameters("x_2", [-6, 6], MixtureModel([TruncatedNormal(2.0, 1.0, -6, 6), TruncatedNormal(-2.0, 1.0, -6, 6)], [0.5, 0.5]))
models = [θ1, θ2]

himmelblau(x1,x2) = (x1^2+x2-11)^2 + (x1+x2^2-7)^2
himmelblau(x) = himmelblau(x[1], x[2])
f_himmelblau(sparams::DummyHimmelblauSystem, x; fx=himmelblau(x)) = fx ≤ sparams.γ

function System.evaluate(sparams::DummyHimmelblauSystem, inputs::Vector; verbose=false, kwargs...)
    verbose && @info "Evaluating dummy Himmelblau system ($inputs)..."
    Y = Vector{Real}(undef, 0)
    for input in inputs
        failure = f_himmelblau(sparams, input)
        push!(Y, failure)
    end
    return Y
end
