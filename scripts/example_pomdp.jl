using Distributed
NUM_PROCS = 1 # 10 # 3
nprocs() < NUM_PROCS && addprocs(NUM_PROCS)

module BetaZero
    using POMDPs
    using Random
    export BeliefMDP
    include("belief_mdp.jl")
end

@everywhere begin
    using Revise
    using BayesianSafetyValidation
    using .BetaZero
    using LightDark
    using ParticleBeliefs
    using ParticleFilters
    using POMDPs
    using POMDPTools
    using LocalApproximationValueIteration
    using LocalFunctionApproximation
    using GridInterpolations
    using SharedArrays
    using BSON


    function input_representation(b::ParticleHistoryBelief{LightDarkState})
        Y = [s.y for s in ParticleFilters.particles(b)]
        μ = mean(Y)
        σ = std(Y)
        return Float32[μ, σ]
    end

    lightdark_belief_reward(pomdp, b, a, bp) = mean(reward(pomdp, s, a) for s in ParticleFilters.particles(b))
    POMDPs.convert_s(::Type{A}, b::ParticleHistoryBelief{LightDarkState}, m) where A<:AbstractArray = eltype(A)[input_representation(b)...]
    POMDPs.convert_s(::Type{ParticleHistoryBelief{LightDarkState}}, b::A, m) where A<:AbstractArray = ParticleHistoryBelief(particles=ParticleCollection(rand(LDNormalStateDist(b[1], b[2]), up.pf.n_init)))

    global USE_LD5 = false

    if USE_LD5
        !@isdefined(LAVI_POLICY_5) && global const LAVI_POLICY_5 = BSON.load(joinpath(@__DIR__, "policy_lavi_ld5_timing.bson"))[:lavi_policy]
        global MODEL = LightDarkPOMDP(; light_loc=5, sigma = y->abs(y - 5)/sqrt(2) + 1e-2, correct_r=10, incorrect_r=-10)
    else
        !@isdefined(LAVI_POLICY_10) && global const LAVI_POLICY_10 = BSON.load(joinpath(@__DIR__, "policy_lavi_ld10_timing.bson"))[:lavi_policy]
        global MODEL = LightDarkPOMDP()
    end

    global const UP = BootstrapFilter(MODEL, 500)

    # NOTE:
    # rare = did not execute stop (got lost)
    # non-rare = did not execute stop or stopped NOT at the goal 0 ± 1
    global RARE_FAILURE = true

    Base.@kwdef mutable struct LightDarkPolicy <: System.SystemParameters
        pomdp::POMDP = MODEL
        updater::Updater = UP
        policy::Policy = USE_LD5 ? LAVI_POLICY_5 : LAVI_POLICY_10
        max_steps::Int = RARE_FAILURE ? 50 : 100
    end

    function System.evaluate(sparams::LightDarkPolicy, inputs::Vector; kwargs...)
        pomdp = sparams.pomdp
        π = sparams.policy
        up = sparams.updater
        max_steps = sparams.max_steps

        # inputs: [[y from uncertainty acq], [y from boundary acq], [y from p(fail) acq]]
        Y = SharedArray{Bool}(length(inputs))
        @sync @distributed for i in eachindex(inputs)
            ys0 = inputs[i]
            s0 = LightDarkState(0, ys0[1])
            ds0 = initialstate(pomdp)
            b0 = initialize_belief(up, ds0)
            history = simulate(HistoryRecorder(max_steps=max_steps), pomdp, π, up, b0, s0)
            g = discounted_reward(history)
            Y[i] = RARE_FAILURE ? g == 0 : g <= 0
        end
        return Y
    end


    function nominal_estimate(sparams::LightDarkPolicy, models::Vector{OperationalParameters}; n=RARE_FAILURE ? 100_000 : 5_000)
        inputs = map(sample->System.generate_input(sparams, sample), rand(models, n))
        Y = SharedArray{Bool}(length(inputs))
        @sync @distributed for i in eachindex(Y)
            Y[i] = System.evaluate(sparams, [inputs[i]])[1]
            @info "$i/$(length(Y)) running estimate = $(mean_and_std(Y[1:i]))"
        end
        @info "Finished computing nominal: $(mean_and_stderr(Y))"
        return Y
    end
end

system_params = LightDarkPolicy()
ds0 = initialstate(system_params.pomdp)
models = [OperationalParameters("initial y-state", [-20, 20], Normal(ds0.mean, ds0.std))]

global nominal_filename = joinpath(@__DIR__, "nominal_lightdark_$(system_params.max_steps)horizon_$(RARE_FAILURE ? "rare" : "nonrare").bson")
global RUN_NOMINAL = false

if RUN_NOMINAL
    @time nominal = nominal_estimate(system_params, models)
    BSON.@save nominal_filename nominal
end

# !@isdefined(nominal) && 
BSON.@load nominal_filename nominal
# nominal = fill(0.0005, 4*10^4)

global LOAD_GP = false

if LOAD_GP
    @info "Loading GP..."
    BSON.@load "gp_lightdark_$(RARE_FAILURE ? "rare" : "nonrare").bson" gp
elseif !RUN_NOMINAL
    global surrogate, weights, X_failures, ml_failure, p_failure, gp, T

    for seed in 1:1
        # N = RARE_FAILURE ? Int(1.5*10^4) : 300 # 3000
        N = RARE_FAILURE ? Int(5000*3) : 300 # 3000
        T = N÷3
        nominalN = RARE_FAILURE ? 4*10^4 : length(nominal)
        surrogate, weights = bayesian_safety_validation(
                                system_params, models;
                                T=T,
                                λ=RARE_FAILURE ? 0.1 : 1, # 0.5,
                                αᵤ=RARE_FAILURE ? 10 : 2, # 10
                                αᵦ=RARE_FAILURE ? 10 : 0,
                                sample_from_acquisitions=[true,true,true],
                                self_normalizing=true,
                                # sample_temperature=RARE_FAILURE ? 0.25 : 1, # 0.8 is nice.
                                sample_temperature=RARE_FAILURE ? 0.5 : 1, # 0.8 is nice.
                                frs_loosening=true,
                                show_plots=false,
                                plot_every=10,
                                show_p_estimates=true,
                                show_num_failures=true,
                                print_p_estimates=true,
                                nominal=nominal[1:nominalN])
                                # nominal=nominal[1:3T])
        X_failures = falsification(surrogate.x, surrogate.y)
        ml_failure = most_likely_failure(surrogate.x, surrogate.y, models)
        p_failure  = p_estimate(surrogate, models; weights)

        gp = surrogate
        BSON.@save "gp_lightdark_$(RARE_FAILURE ? "rare" : "nonrare")_$seed.bson" gp
        BSON.@save "weights_lightdark_$(RARE_FAILURE ? "rare" : "nonrare")_$seed.bson" weights

        compute_metrics(gp, models, system_params; weights)

        plot1d(gp, models) |> display
        #==#
        num_samples, p_estimates, p_estimate_confs = recompute_p_estimates(gp, models; weights)
        plot_p_estimates(num_samples, p_estimates, p_estimate_confs; nominal=nominal[1:nominalN], full_nominal=false, gpy=gp.y, scale=1.5, logscale=false)
        # plot_p_estimates(1:N, fill(0.0,N), fill(NaN, N); nominal=nominal[1:4*10^4], full_nominal=true)
        #==#
    end
end
