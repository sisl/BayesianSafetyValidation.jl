"""
Parametric space consisting of the discrete `range` and a sampleable `distribution`.
"""
mutable struct OperationalParameters
    name::String
    range::AbstractArray
    distribution::Sampleable
end

Base.rand(models::Vector{OperationalParameters}) = [rand(m.distribution) for m in models]
Base.rand(models::Vector{OperationalParameters}, n::Int) = [rand(models) for _ in 1:n]
Distributions.pdf(model::OperationalParameters, x::Number) = pdf(model.distribution, x)
Distributions.pdf(models::Vector{OperationalParameters}, x::Vector) = prod([pdf(m.distribution, x[i]) for (i,m) in enumerate(models)])
Distributions.pdf(models::Vector{OperationalParameters}, x::SubArray) = pdf(models, [x...])
