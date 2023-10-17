"""
Main algorithm: iterative sample points and re-fit the Gaussian process surrogate model.
"""
function bayesian_safety_validation(sparams, models;
                            gp=nothing,
                            gp_args=(ν=1/2, ll=-0.1, lσ=-0.1, opt=false),
                            T=1,
                            λ=0.1, # UCB
                            αᵤ=Inf, # decay rate for uncertainty exploration (Inf disables operational model influence)
                            αᵦ=1, # decay rate for boundary refinement
                            seed=0,
                            initialize_system=false,
                            reset_system=true,
                            hard_is_estimate=true,
                            self_normalizing=false,
                            sample_temperature=1,
                            show_acquisition_plots=false,
                            show_plots=false,
                            show_combined_plot=false,
                            show_tight_combined_plot=false,
                            show_p_estimates=false,
                            print_p_estimates=false,
                            hide_model_after_first=false,
                            only_plot_last=false,
                            latex_labels=true,
                            show_alert=false,
                            acquisitions_to_run=[1,2,3], # 1 = uncertainty exploration, 2 = boundary refinement, 3 = failure region sampling
                            sample_from_acquisitions=[false,false,true], # uncertainty, boundary, failure region.
                            probability_valued_system=true,
                            skip_if_no_failures=false,
                            save_plots=false,
                            save_plots_svg=false,
                            plots_dir="plots",
                            samples_per_batch=1,
                            single_failure_mode=false,
                            refit_every_point=false,
                            m=200, # alias for `input_discretization_steps`
                            input_discretization_steps=m,
                            d=500, # alias for `p_estimate_discretization_steps`
                            p_estimate_discretization_steps=d,
                            match_original=false,
                            kwargs...)
    try
        Random.seed!(seed)
        num_dimensions = length(models)
        W = Float64[] # self-normalizing weights

        if show_p_estimates
            p_estimates = []
        end

        if isnothing(gp)
            initialize_system && System.initialize()
            reset_system && System.reset(sparams)

            inputs = []
            X = Matrix{Float64}(undef, num_dimensions, 0)
            Y = Float64[]

            #= Surrogate modeling =#
            gp = gp_fit(X, Y; gp_args...)
            t_offset = 0
        else
            X = gp.x
            Y = inverse.(gp.y)
            t_offset = length(Y) ÷ length(acquisitions_to_run)
        end

        y = gp_output(gp, models, num_steps=input_discretization_steps)
        if show_plots && !show_acquisition_plots && !(show_combined_plot || show_tight_combined_plot)
            # initial plot without observations
            if num_dimensions == 1
                display(plot1d(gp, models; num_steps=input_discretization_steps))
            else
                display(plot_soft_boundary(gp, models; num_steps=input_discretization_steps))
            end
        end

        if (save_plots || save_plots_svg) && !isdir(plots_dir)
            mkdir(plots_dir)
        end

        sample_from_acquisition = false

        function append_sample(X, next_point; t)
            sample = next_point[1:num_dimensions]
            X = hcat(X, sample)
            input = System.generate_input(sparams, sample; models, subdir=t, kwargs...)
            push!(inputs, input)
            return X
        end

        #= Failure boundary refinement =#
        for t in (1+t_offset):(T+t_offset)
            inputs = []
            acq_plts = []

            if !refit_every_point
                # Precompute GP prediction to pass to each acquisition function (faster)
                F̂ = gp_output(gp, models; num_steps=input_discretization_steps, f=predict_f_vec)

                # Precompute operational likelihood over domain (faster)
                P = p_output(models; num_steps=input_discretization_steps)
            end

            println("\n", "-"^40)
            for a in acquisitions_to_run
                if refit_every_point
                    inputs = []
                    # Recompute
                    F̂ = gp_output(gp, models; num_steps=input_discretization_steps, f=predict_f_vec)
                    P = p_output(models; num_steps=input_discretization_steps)
                end
                @info "Refinement iteration $t (acquisition $a)"
                if a == 1
                    if any(Y .== 1) && single_failure_mode
                        @warn "Skipping exploration (single failure mode found)."
                        continue # skip exploration when a failure is already found (NOTE: only in `single_failure_mode`)
                    end
                    acq = (μ_σ²,p)->uncertainty_acquisition(μ_σ², p; t, α=αᵤ)
                    sample_from_acquisition = sample_from_acquisitions[1]
                elseif a == 2
                    if !any(Y .== 1) && skip_if_no_failures
                        @warn "No failures found, skipping acquisition for failure boundary."
                        continue
                    end
                    acq = (μ_σ²,p)->boundary_acquisition(μ_σ², p; λ, t, α=αᵦ)
                    sample_from_acquisition = sample_from_acquisitions[2]
                elseif a == 3
                    if !any(Y .== 1) && skip_if_no_failures
                        @warn "No failures found, skipping acquisition for failure distribution."
                        continue
                    end
                    acq = (μ_σ²,p)->failure_region_sampling_acquisition(μ_σ², p; λ, probability_valued_system)
                    sample_from_acquisition = sample_from_acquisitions[3]
                else
                    error("No acquisition function defined for a=$a")
                end

                if sample_from_acquisition
                    next_points_and_weight = sample_next_point(y, F̂, P, models; acq, n=samples_per_batch, τ=sample_temperature, match_original, return_weight=self_normalizing)
                    if self_normalizing
                        next_points, weight = next_points_and_weight
                    else
                        next_points = next_points_and_weight
                    end
                    for (i,next_point) in enumerate(next_points)
                        X = append_sample(X, next_point; t)
                        if self_normalizing
                            push!(W, weight[i])
                        end
                    end
                else
                    next_point_and_weight = get_next_point(y, F̂, P, models; acq, match_original, return_weight=self_normalizing)
                    if self_normalizing
                        if !all(sample_from_acquisitions)
                            error("Please make sure all acquisition functions are sampled from to use Self-normalzing importance sampling (see `sample_from_acquisitions`)")
                        end
                        next_point, weight = next_point_and_weight
                        push!(W, weight)
                    else
                        next_point = next_point_and_weight
                    end
                    X = append_sample(X, next_point; t)
                end


                if show_acquisition_plots || show_combined_plot || show_tight_combined_plot
                    plt_gp = plot_soft_boundary(gp, models; num_steps=input_discretization_steps)
                    next_point_ms = (show_combined_plot || show_tight_combined_plot) ? 3 : 5
                    plt_acq = plot_acquisition(y, F̂, P, models; acq, zero_white=sample_from_acquisition, given_next_point=sample_from_acquisition ? next_points[1] : next_point, ms=next_point_ms, tight=show_tight_combined_plot)
                    if (show_combined_plot || show_tight_combined_plot)
                        push!(acq_plts, plt_acq)
                    elseif show_acquisition_plots
                        plt = plot(plt_gp, plt_acq, size=(750,300), margin=3Plots.mm)
                        try
                            display(plt)
                        catch err
                            @warn err
                        end
                        if save_plots
                            plt_filename = joinpath(plots_dir, "plot-$(t)-$(a)")
                            savefig_dense(plt, plt_filename)
                        end
                        if save_plots_svg
                            plt_filename = joinpath(plots_dir, "plot-$(t)-$(a).svg")
                            savefig(plt, plt_filename)
                        end
                    end
                end

                if refit_every_point
                    Y′ = System.evaluate(sparams, inputs; subdir=t, kwargs...)
                    Y = vcat(Y, Y′)
                    gp = gp_fit(X, Y; gp_args...)
                end
            end

            # TODO: Separate function.
            if (show_combined_plot || show_tight_combined_plot) && ((only_plot_last && t == T) || !only_plot_last)
                acq_titlefontsize = show_tight_combined_plot ? 12 : 10
                acq_plts[1] = plot(acq_plts[1], title="uncertainty exploration", titlefontsize=acq_titlefontsize)
                acq_plts[2] = plot(acq_plts[2], title="boundary refinement", titlefontsize=acq_titlefontsize)
                acq_plts[3] = plot(acq_plts[3], title="failure region sampling", titlefontsize=acq_titlefontsize)

                if show_tight_combined_plot
                    plt = plot_combined(gp, models, sparams; num_steps=input_discretization_steps, surrogate=true, show_data=true, title="surrogate", titlefontsize=show_tight_combined_plot ? acq_titlefontsize : 18, tight=show_tight_combined_plot, acq_plts, hide_model=hide_model_after_first && t > 1, add_phantom_point=true, latex_labels)
                else
                    plt_surrogate_models = plot_combined(gp, models, sparams; num_steps=input_discretization_steps, surrogate=true, show_data=true, title="surrogate", tight=show_tight_combined_plot, latex_labels)
                    plt_acquisitions = plot(acq_plts..., layout=(1,3))
                    plt = plot(plt_surrogate_models, plt_acquisitions, layout=@layout([a{0.8h}; b]), size=(750, 650))
                end
                try
                    display(plt)
                catch err
                    @warn err
                end
                acq_plts = []
                if save_plots
                    plt_filename = joinpath(plots_dir, "plot-combined-$t")
                    savefig_dense(plt, plt_filename)
                end
                if save_plots_svg
                    plt_filename = joinpath(plots_dir, "plot-combined-$t.svg")
                    savefig(plt, plt_filename)
                end
            end

            if !refit_every_point
                Y′ = System.evaluate(sparams, inputs; subdir=t, kwargs...)
                Y = vcat(Y, Y′)
                gp = gp_fit(X, Y; gp_args...)
            end

            if show_plots && !show_acquisition_plots
                # already show this above
                if num_dimensions == 1
                    plot1d(gp, models; num_steps=input_discretization_steps) |> display
                else
                    plot_soft_boundary(gp, models; num_steps=input_discretization_steps) |> display
                end
            end

            if show_p_estimates || print_p_estimates
                if self_normalizing
                    p_fail = is_self_normalizing(gp, W)
                else
                    p_fail = p_estimate(gp, models; num_steps=p_estimate_discretization_steps, hard=hard_is_estimate)
                end
                if print_p_estimates
                    @info "P(fail) = $p_fail"
                end
                if show_p_estimates
                    push!(p_estimates, p_fail)
                    plot_p_estimates(3:3:3t, p_estimates) |> display
                end
            end
        end

        if show_plots
            if num_dimensions == 1
                plot1d(gp, models; num_steps=input_discretization_steps) |> display
            else
                plot_soft_boundary(gp, models; num_steps=input_discretization_steps) |> display
            end
        end

        if show_p_estimates || print_p_estimates
            if print_p_estimates
                if self_normalizing
                    p_fail = is_self_normalizing(gp, W)
                else
                    p_fail = p_estimate(gp, models; num_steps=p_estimate_discretization_steps, hard=hard_is_estimate)
                end
                @info "P(fail) = $p_fail"
            end
            if show_p_estimates
                plot_p_estimates(3:3:3T, p_estimates) |> display
            end
        end

        if show_alert
            alert("GP iteration finished.")
        end

        if self_normalizing
            p_fail = is_self_normalizing(gp, W)
        else
            p_fail = p_estimate(gp, models; num_steps=p_estimate_discretization_steps, hard=hard_is_estimate)
        end
        @info "p(fail) estimate = $p_fail"

        if self_normalizing
            return gp, W
        else
            return gp
        end
    catch err
        alert("Error in `bayesian_safety_validation`!")
        rethrow(err)
    end
end


"""
Run a single input sample through the system, re-fit suurogate model.
"""
function run_single_sample(gp, models, sparams, sample; subdir="test")
    input = System.generate_input(sparams, sample; models, subdir)
    inputs = [input]
    X = hcat(gp.x, sample)
    Y′ = System.evaluate(sparams, inputs; subdir=subdir)
    Y = vcat(gp.y, Y′)
    gp = gp_fit(X, Y)
    display(plot_soft_boundary(gp, models))
    return gp
end
