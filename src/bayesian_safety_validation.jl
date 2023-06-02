"""
Main algorithm: iterative sample points and re-fit the Gaussian process surrogate model.
"""
function bayesian_safety_validation(sparams, models;
                            gp=nothing,
                            T=1,
                            λ=0.1, # UCB
                            seed=0,
                            initialize_system=false,
                            reset_system=true,
                            show_acquisition_plots=false,
                            show_plots=false,
                            show_combined_plot=false,
                            show_tight_combined_plot=false,
                            hide_model_after_first=false,
                            only_plot_last=false,
                            latex_labels=true,
                            show_alert=false,
                            acquisitions_to_run=[1,2,3], # 1 = uncertainty exploration, 2 = boundary refinement, 3 = failure region sampling
                            probability_valued_system=true,
                            skip_if_no_failures=false,
                            save_plots=false,
                            save_plots_svg=false,
                            plots_dir="plots",
                            samples_per_batch=1,
                            single_failure_mode=false,
                            refit_every_point=false,
                            input_discritization_steps=200,
                            p_estimate_discritization_steps=500,
                            match_original=false,
                            kwargs...)
    try
        Random.seed!(seed)

        if isnothing(gp)
            initialize_system && System.initialize()
            reset_system && System.reset(sparams)

            inputs = []
            X = Matrix{Float64}(undef, length(models), 0)
            Y = Float64[]

            #= Surrogate modeling =#
            gp = gp_fit(X, Y)
            t_offset = 0
        else
            X = gp.x
            Y = inverse.(gp.y)
            t_offset = length(Y) ÷ length(acquisitions_to_run)
        end

        y = gp_output(gp, models, num_steps=input_discritization_steps)
        if show_plots && !show_acquisition_plots && !(show_combined_plot || show_tight_combined_plot)
            display(plot_soft_boundary(gp, models))
        end

        if (save_plots || save_plots_svg) && !isdir(plots_dir)
            mkdir(plots_dir)
        end

        sample_from_acquisition = false

        function append_sample(X, next_point; t)
            sample = next_point[1:size(X)[1]]
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
                F̂ = gp_output(gp, models; num_steps=input_discritization_steps, f=predict_f_vec)

                # Precompute operational likelihood over domain (faster)
                P = p_output(models; num_steps=input_discritization_steps)
            end

            println("\n", "-"^40)
            for a in acquisitions_to_run
                if refit_every_point
                    inputs = []
                    # Recompute
                    F̂ = gp_output(gp, models; num_steps=input_discritization_steps, f=predict_f_vec)
                    P = p_output(models; num_steps=input_discritization_steps)
                end
                @info "Refinement iteration $t (acquisition $a)"
                if a == 1
                    if any(Y .== 1) && single_failure_mode
                        @warn "Skipping exploration (single failure mode found)."
                        continue # skip exploration when a failure is already found (NOTE: only in `single_failure_mode`)
                    end
                    acq = uncertainty_acquisition
                    sample_from_acquisition = false
                elseif a == 2
                    if !any(Y .== 1) && skip_if_no_failures
                        @warn "No failures found, skipping acquisition for failure boundary."
                        continue
                    end
                    acq = (μ_σ²,p)->boundary_acquisition(μ_σ²,p; λ, t)
                    sample_from_acquisition = false
                elseif a == 3
                    if !any(Y .== 1) && skip_if_no_failures
                        @warn "No failures found, skipping acquisition for failure distribution."
                        continue
                    end
                    acq = (μ_σ²,p)->failure_region_sampling_acquisition(μ_σ²,p; λ, probability_valued_system)
                    sample_from_acquisition = true
                else
                    error("No acquisition function defined for a=$a")
                end

                if sample_from_acquisition
                    next_points = sample_next_point(y, F̂, P, models; acq, n=samples_per_batch, match_original)
                    for next_point in next_points
                        X = append_sample(X, next_point; t)
                    end
                else
                    next_point = get_next_point(y, F̂, P, models; acq, match_original)
                    X = append_sample(X, next_point; t)
                end


                if show_acquisition_plots || show_combined_plot || show_tight_combined_plot
                    plt_gp = plot_soft_boundary(gp, models)
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
                    gp = gp_fit(X, Y)
                end
            end

            # TODO: Separate function.
            if (show_combined_plot || show_tight_combined_plot) && ((only_plot_last && t == T) || !only_plot_last)
                acq_titlefontsize = show_tight_combined_plot ? 12 : 10
                acq_plts[1] = plot(acq_plts[1], title="uncertainty exploration", titlefontsize=acq_titlefontsize)
                acq_plts[2] = plot(acq_plts[2], title="boundary refinement", titlefontsize=acq_titlefontsize)
                acq_plts[3] = plot(acq_plts[3], title="failure region sampling", titlefontsize=acq_titlefontsize)

                if show_tight_combined_plot
                    plt = plot_combined(gp, models, sparams; surrogate=true, show_data=true, title="surrogate", titlefontsize=show_tight_combined_plot ? acq_titlefontsize : 18, tight=show_tight_combined_plot, acq_plts, hide_model=hide_model_after_first && t > 1, add_phantom_point=true, latex_labels)
                else
                    plt_surrogate_models = plot_combined(gp, models, sparams; surrogate=true, show_data=true, title="surrogate", tight=show_tight_combined_plot, latex_labels)
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
                gp = gp_fit(X, Y)
            end

            if show_plots && !show_acquisition_plots
                # already show this above
                plot_soft_boundary(gp, models) |> display
            end
        end

        if show_plots
            plot_soft_boundary(gp, models) |> display
        end

        if show_alert
            alert("GP iteration finished.")
        end

        @info "p(fail) estimate = $(p_estimate(gp, models; num_steps=p_estimate_discritization_steps))"

        return gp
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
