@with_kw mutable struct DummyBoothSystem <: System.SystemParameters
    γ = 200
end

system_params = DummyBoothSystem()
θ1 = OperationalParameters("x", [-10, 5], TruncatedNormal(-10, 1.5, -10, 5))
θ2 = OperationalParameters("y", [-10, 5], Normal(-2.5, 1))
models = [θ1, θ2]

function System.reset(sparams::DummyBoothSystem) end

function System.initialize(; kwargs...) end

function System.generate_input(sparams::DummyBoothSystem, sample::Vector; kwargs...)
    return sample
end

booth(x,y) = (x+2y-7)^2 + (2x+y-5)^2
booth(x) = booth(x[1], x[2])
f_booth(sparams::DummyBoothSystem, x) =  booth(x) ≤ sparams.γ

function System.evaluate(sparams::DummyBoothSystem, inputs::Vector; kwargs...)
    @info "Evaluating dummy linear system ($inputs)..."
    Y = Vector{Bool}(undef, 0)
    for input in inputs
        failure = f_booth(sparams, input)
        push!(Y, failure)
    end
    return Y
end