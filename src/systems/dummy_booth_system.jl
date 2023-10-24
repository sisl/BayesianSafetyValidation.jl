@with_kw mutable struct DummyBoothSystem <: System.SystemParameters
    γ = 200
end

system_params = DummyBoothSystem()
θ1 = OperationalParameters("x_1", [-10, 5], TruncatedNormal(-10, 1.5, -10, 5))
θ2 = OperationalParameters("x_2", [-10, 5], Normal(-2.5, 1))
models = [θ1, θ2]

booth(x1,x2) = (x1 + 2x2 - 7)^2 + (2x1 + x2 - 5)^2
booth(x) = booth(x[1], x[2])
f_booth(sparams::DummyBoothSystem, x) =  booth(x) ≤ sparams.γ

function System.evaluate(sparams::DummyBoothSystem, inputs::Vector; verbose=false, kwargs...)
    verbose && @info "Evaluating dummy Booth system ($inputs)..."
    Y = Vector{Bool}(undef, 0)
    for input in inputs
        failure = f_booth(sparams, input)
        push!(Y, failure)
    end
    return Y
end
