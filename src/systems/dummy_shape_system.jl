using FileIO
using ImageIO
using Suppressor

###########################################################
## Setup the dummy shape (aircraft icon) system.
###########################################################
function loadsprite()
    filename = joinpath(@__DIR__, "..", "..", "media", "aircraft.png")
    S = @suppress load(filename)
    S = Gray.(S)
    return S
end

@with_kw struct DummyShapeParams <: System.SystemParameters
    S = loadsprite()
end
system_params = DummyShapeParams()

θ1 = OperationalParameters("x", [1, 120], Distributions.Uniform(1, 120))
θ2 = OperationalParameters("y", [1, 120], Distributions.Uniform(1, 120))
models = [θ1, θ2]

###########################################################
## System interface implementation.
###########################################################

function System.evaluate(sparams::DummyShapeParams, inputs::Vector; verbose=false, kwargs...)
    verbose && @info "Evaluating dummy shape ($inputs)..."
    Y = Vector{Float64}(undef, 0)
    S = sparams.S
    for input in inputs
        x, y = input
        j, i = trunc(Int, 10x), trunc(Int, size(S,2) + 1 - 10y) # note i-j flip
        # if S[i,j] == Gray(1) # black
        #     failure = 0
        # elseif S[i,j] == Gray(0) # white
        #     failure = 0.25
        # else
        #     failure = 1
        # end
        # failure = Float64(S[i,j].val)
        # failure = S[i,j] != Gray(0)
        failure = S[i,j] == Gray(0)
        push!(Y, failure)
    end
    return Y
end