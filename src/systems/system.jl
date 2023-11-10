module System

export
    SystemParameters,
    reset,
    initialize,
    generate_input,
    evaluate


"""
Abstract base type for parameters used by the system under test.
"""
abstract type SystemParameters end


"""
Interface function to reset the system under test.
"""
function reset(sparams::SystemParameters; kwargs...) end


"""
Interface function to initialize the system under test.
"""
function initialize(sparams::SystemParameters; kwargs...) end


"""
Interface function to generate input to the system based on the selected parametric sample,
e.g., take an image and return the parameters used to generate that image.
"""
generate_input(sparams::SystemParameters, sample::Vector; kwargs...)::Vector = sample # Default: pass-through


"""
Interface function to call/evaluate the system under test (SUT) given the generated input.
Returns a boolean `true` if the system failed.
"""
function evaluate(sparams::SystemParameters, inputs::Vector; kwargs...)::Vector{Bool} end


end # module
