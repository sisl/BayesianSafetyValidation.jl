"""
Parametric space consisting of the discrete `range` and a sampleable `distribution`.
"""
mutable struct OperationalParameters
    name::String
    range::AbstractArray
    distribution::Sampleable
end
