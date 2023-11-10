@with_kw mutable struct DummyLinearParameters <: System.SystemParameters
    failure_point_x = [5]
    failure_point_y = [5]
end

system_params = DummyLinearParameters(failure_point_x = [2])
θ1 = OperationalParameters("x", [0, 10], Normal(5, 1))
θ2 = OperationalParameters("y", [0, 10], Normal(5, 1))
models = [θ1, θ2]

function System.evaluate(sparams::DummyLinearParameters, inputs::Vector; kwargs...)
    @info "Evaluating dummy linear system ($inputs)..."
    Y = Vector{Bool}(undef, 0)
    x_vec = sparams.failure_point_x
    y_vec = sparams.failure_point_y
    for input in inputs
        failure = false
        for i in eachindex(x_vec)
            x_fail = x_vec[i]
            y_fail = y_vec[i]
            x, y = input
            local_failure = (x ≤ x_fail)
            failure = failure || local_failure
        end
        push!(Y, failure)
    end
    return Y
end
