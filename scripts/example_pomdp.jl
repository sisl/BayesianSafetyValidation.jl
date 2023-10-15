using Revise
using BayesianSafetyValidation
using BetaZero
using LightDark
using ParticleBeliefs
using ParticleFilters
using POMDPs
using POMDPTools
using LocalApproximationValueIteration
using LocalFunctionApproximation
using GridInterpolations
using BSON

function BetaZero.input_representation(b::ParticleHistoryBelief{LightDarkState})
    Y = [s.y for s in ParticleFilters.particles(b)]
    μ = mean(Y)
    σ = std(Y)
    return Float32[μ, σ]
end

lightdark_belief_reward(pomdp, b, a, bp) = mean(reward(pomdp, s, a) for s in ParticleFilters.particles(b))
POMDPs.convert_s(::Type{A}, b::ParticleHistoryBelief{LightDarkState}, m::BeliefMDP) where A<:AbstractArray = eltype(A)[BetaZero.input_representation(b)...]
POMDPs.convert_s(::Type{ParticleHistoryBelief{LightDarkState}}, b::A, m::BeliefMDP) where A<:AbstractArray = ParticleHistoryBelief(particles=ParticleCollection(rand(LDNormalStateDist(b[1], b[2]), up.pf.n_init)))

!@isdefined(LAVI_POLICY) && global const LAVI_POLICY = BSON.load("policy_lavi_ld10_timing.bson")[:lavi_policy]
global const MODEL = LightDarkPOMDP()
global const UP = BootstrapFilter(MODEL, 500)

@with_kw mutable struct LightDarkPolicy <: System.SystemParameters
    pomdp::POMDP = MODEL
    updater::Updater = UP
    policy::Policy = LAVI_POLICY
    max_steps::Int = 100
end

System.generate_input(sparams::LightDarkPolicy, sample::Vector; kwargs...) = sample # pass-through

function System.reset(::LightDarkPolicy) end
function System.initialize(; kwargs...) end

function System.evaluate(sparams::LightDarkPolicy, inputs::Vector; kwargs...)
    pomdp = sparams.pomdp
    π = sparams.policy
    up = sparams.updater
    max_steps = sparams.max_steps

    # inputs: [[y from uncertainty acq], [y from boundary acq], [y from p(fail) acq]]
    Y = Vector{Bool}(undef, length(inputs))
    for (i,ys0) in enumerate(inputs)
        s0 = LightDarkState(0, ys0[1])
        ds0 = initialstate_distribution(pomdp)
        b0 = initialize_belief(up, ds0)
        history = simulate(HistoryRecorder(max_steps=max_steps), pomdp, π, up, b0, s0)
        g = discounted_reward(history)
        Y[i] = g <= 0
    end
    return Y
end

system_params = LightDarkPolicy()
ds0 = initialstate_distribution(system_params.pomdp)
model = [OperationalParameters("initial y-state", [-20, 20], Normal(ds0.mean, ds0.std))]

surrogate  = bayesian_safety_validation(system_params, model; T=30)
X_failures = falsification(surrogate.x, surrogate.y)
ml_failure = most_likely_failure(surrogate.x, surrogate.y, model)
p_failure  = p_estimate(surrogate, model)

gp = surrogate

BSON.@save "gp_lightdark.bson" gp

plot1d(surrogate, model)
