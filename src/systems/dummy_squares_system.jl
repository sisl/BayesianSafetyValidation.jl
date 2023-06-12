###########################################################
## Setup the "dummy squares" system.
###########################################################
@with_kw mutable struct DummyParameters <: System.SystemParameters
    failure_point = [[2,2]]
    failure_radius = [2]
end

@enum DummySystemSquaresConfig SingleSquare MultipleSquares SingleSquare3d
@enum DummySystemSquaresModels NormalSquareModels UniformSquareModels UniformSquareModels3d

DUMMY_SYSTEM_CONFIG = MultipleSquares
DUMMY_SYSTEM_MODELS = NormalSquareModels

if DUMMY_SYSTEM_CONFIG == SingleSquare
    system_params = DummyParameters(failure_point=[[3,3]], failure_radius=[1])
elseif DUMMY_SYSTEM_CONFIG == SingleSquare3d
    system_params = DummyParameters(failure_point=[[1,1,1]], failure_radius=[1])
else
    system_params = DummyParameters(failure_point=[[2, 2], [8, 8]], failure_radius=[1, 1/2])
end

if DUMMY_SYSTEM_MODELS == NormalSquareModels
    θ1 = OperationalParameters("x_1", [0, 10], Normal(5, 1))
    θ2 = OperationalParameters("x_2", [0, 10], Normal(5, 1))
    models = [θ1, θ2]
elseif DUMMY_SYSTEM_MODELS == UniformSquareModels3d
    θ1 = OperationalParameters("x_1", [0, 5], Uniform(0, 5))
    θ2 = OperationalParameters("x_2", [0, 5], Uniform(0, 5))
    θ3 = OperationalParameters("x_3", [0, 5], Uniform(0, 5))
    models = [θ1, θ2, θ3]
else
    θ1 = OperationalParameters("x_1", [0, 10], Uniform(0, 10))
    θ2 = OperationalParameters("x_2", [0, 10], Uniform(0, 10))
    models = [θ1, θ2]
end



###########################################################
## System interface implementation.
###########################################################

function System.reset(sparams::DummyParameters) end


function System.initialize(; kwargs...) end


function System.generate_input(sparams::DummyParameters, sample::Vector; kwargs...)
    return sample # pass-through
end


function System.evaluate(sparams::DummyParameters, inputs::Vector; verbose=false, kwargs...)
    verbose && @info "Evaluating dummy system ($inputs)..."
    Y = Vector{Bool}(undef, 0)
    C_vec = sparams.failure_point
    r_vec = sparams.failure_radius
    for input in inputs
        failure = false
        for i in eachindex(C_vec)
            C = C_vec[i]
            r = r_vec[i]
            local_failure = true
            for d in 1:length(C)
                local_failure &= C[d] - r <= input[d] <= C[d] + r
            end
            failure = failure || local_failure
        end
        push!(Y, failure)
    end
    return Y
end


function dummy_true_pfail(sparams, models)
    area = prod([model.range[end] - model.range[1] for model in models])
    # simplifying assumption: in bounds and non-overlapping
    ndim = length(models)
    fregion = sum((2*sparams.failure_radius).^ndim)
    return fregion / area
end


function dummy_error(gp, sparams, models)
    p_estimate(gp, models) - dummy_true_pfail(sparams, models)
end
