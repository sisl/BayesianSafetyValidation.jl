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


function nominal_estimate(sparams::LightDarkPolicy, models::Vector{OperationalParameters}; n=1000)
    inputs = map(sample->System.generate_input(sparams, sample), rand(models, n))
    Y = Vector{Bool}(undef, length(inputs))
    for i in eachindex(Y)
        Y[i] = System.evaluate(sparams, [inputs[i]])[1]
        @info "$i/$(length(Y)) running estimate = $(mean_and_std(Y[1:i]))"
    end
    return mean_and_std(Y)
end

#=
julia> @time nominal_estimate(system_params, models)
126.3 seconds
0.155 ± 0.363
=#
nominal = [0.155, 0.363/sqrt(200)]

system_params = LightDarkPolicy()
ds0 = initialstate_distribution(system_params.pomdp)
models = [OperationalParameters("initial y-state", [-20, 20], Normal(ds0.mean, ds0.std))]

# gp_args = (ν=1/2, ll=log(0.1), lσ=log(10))
# gp_args = (ν=1/2, ll=log(0.05), lσ=log(0.05))
# gp_args = (ν=1/2, ll=log(0.01), lσ=log(0.1))
# gp_args = (ν=1/2, ll=log(0.05), lσ=log(0.1))
# gp_args = (ν=1/2, ll=log(0.05), lσ=log(1))

# gp_args = (ν=1/2, ll=log(1/2), lσ=log(4))
# gp_args = (ν=1/2, ll=log(1/2), lσ=log(1))
# gp_args = (ν=5/2, ll=log(1), lσ=log(1))

# gp_args = (ν=1/2, ll=log(1), lσ=log(1))
pfail_args = (m=10_000, d=10_000, hard=false)

# if false
    surrogate, weights = bayesian_safety_validation(
                            system_params, models;
                            T=40,
                            m=pfail_args.m, # important
                            d=pfail_args.d, # important
                            λ=10, # important (was 10, 2)
                            αᵤ=Inf, # important (was 1e-8...) disabled with Inf (Inf was good <---!)
                            # TODO: Try αᵤ larger numbers +100
                            αᵦ=1, # important (was 0.1...)
                            # gp_args,
                            hard_is_estimate=pfail_args.hard, # important
                            # sample_from_acquisitions=[false,false,true],
                            # sample_from_acquisitions=[false,false,true],
                            sample_from_acquisitions=[true,true,true],
                            self_normalizing=true,
                            sample_temperature=0.25, # focus towards maximum
                            show_plots=true,
                            show_p_estimates=false,
                            print_p_estimates=true)
    X_failures = falsification(surrogate.x, surrogate.y)
    ml_failure = most_likely_failure(surrogate.x, surrogate.y, models)
    p_failure  = p_estimate(surrogate, models; num_steps=pfail_args.m, hard=pfail_args.hard)

    gp = surrogate
    BSON.@save "gp_lightdark.bson" gp

    compute_metrics(gp, models, system_params; weights)

    # iterations, p_estimates = recompute_p_estimates(gp, models; gp_args, step=1)
    # plot_p_estimates(iterations, p_estimates)
# end
