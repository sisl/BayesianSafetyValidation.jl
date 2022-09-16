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
function reset(; kwargs...) end


"""
Interface function to initialize the system under test.
"""
function initialize(; kwargs...) end


"""
Interface function to generate input to the system based on the selected parametric sample.
"""
function generate_input(sample::Vector; kwargs...)::Vector end


"""
Interface function to call/evaluate the system under test (SUT) given the generated input.
Returns a boolean `true` if the system failed.
"""
function evaluate(inputs::Vector; kwargs...)::Vector{Bool} end


end # module
