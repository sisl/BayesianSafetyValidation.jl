"""
Main algorithm: iterative sample points and re-fit the Gaussian process surrogate model.
"""
function iteratively_sample(sparams, models;
                            gp=nothing,
                            M=1,
                            seed=0,
                            initialize_system=false,
                            reset_system=true,
                            show_acquisition=true,
                            show_plots=false,
                            show_combined_plot=false,
                            show_alert=false,
                            combined_acquisitions=false,
                            alternate_acquisitions=true,
                            skip_if_no_failures=false,
                            initialize_corners=false,
                            save_plots=false,
                            plots_dir="plots",
                            samples_per_batch=1,
                            single_failure_mode=false,
                            kwargs...)
    try
        Random.seed!(seed)
        gp_args = (ll=1.0, lσ=0.5, ν=1/2)

        if isnothing(gp)
            initialize_system && System.initialize()
            reset_system && System.reset(sparams)

            X = Matrix{Float64}(undef, 2, 0)

            if initialize_corners
                #= Initial space sampling =#
                inputs = []
                for param1 in models[1].range # e.g., distance
                    for param2 in models[2].range # e.g., slope
                        sample = [param1, param2]
                        X = hcat(X, sample)
                        input = System.generate_input(sparams, sample; models, kwargs...)
                        push!(inputs, input)
                    end
                end

                Y = System.evaluate(sparams, inputs; kwargs...)
            else
                Y = Bool[]
            end
            #= Surrogate modeling =#
            gp = gp_fit(X, Y; gp_args...)
            m_offset = 0
        else
            X = gp.x
            Y = gp.y
            m_offset = length(Y) - prod(model->length(model.range), models)
        end

        y = gp_output(gp, models)
        λ = 0.1 # UCB
        if show_plots && !show_acquisition && !show_combined_plot
            display(plot_soft_boundary(gp, models))
        end

        if save_plots && !isdir(plots_dir)
            mkdir(plots_dir)
        end

        sample_from_acquisition = false

        function append_sample(X, next_point; m)
            sample = next_point[1:2]
            X = hcat(X, sample)
            input = System.generate_input(sparams, sample; models, subdir=m, kwargs...)
            push!(inputs, input)
            return X
        end

        #= Failure boundary refinement =#
        t = m_offset + 1
        for m in (1+m_offset):(M+m_offset)
            inputs = []
            acq_plts = []
            for a in 1:(alternate_acquisitions ? 3 : 1)
                @info "Refinement iteration $m (acquisition $a)"
                if a == 1
                    if !any(Y .== 1) && skip_if_no_failures
                        @warn "No failures found, skipping acquisition for failure distribution."
                        continue
                    end
                    acq = (gp,x)->operational_acquisition(gp, x, models; λ, t)
                    sample_from_acquisition = true
                elseif a == 2
                    if !any(Y .== 1) && skip_if_no_failures
                        @warn "No failures found, skipping acquisition for failure boundary."
                        continue
                    end
                    acq = (gp,x)->boundary_acquisition(gp, x, models; λ, t)
                    if combined_acquisitions
                        acq_explore = (gp,x)->uncertainty_acquisition(gp, x, models; t)
                    end
                    sample_from_acquisition = false
                else
                    if any(Y .== 1) && single_failure_mode
                        @warn "Skipping exploration (single failure mode found)."
                        continue # skip exploration when a failure is already found (NOTE: only in `single_failure_mode`)
                    end
                    acq = (gp,x)->uncertainty_acquisition(gp, x, models; t)
                    sample_from_acquisition = false
                end


                if sample_from_acquisition
                    next_points = sample_next_point(gp, y, models; acq, n=samples_per_batch)
                    for next_point in next_points
                        X = append_sample(X, next_point; m)
                    end
                else
                    next_point = get_next_point(gp, y, models; acq, acq_explore=combined_acquisitions ? acq_explore : nothing)
                    X = append_sample(X, next_point; m)
                end


                if show_acquisition || show_combined_plot
                    plt_gp = plot_soft_boundary(gp, models)
                    next_point_ms = show_combined_plot ? 3 : 5
                    plt_acq = plot_acquisition(gp, y, models; acq, acq_explore=combined_acquisitions ? acq_explore : nothing, given_next_point=sample_from_acquisition ? next_points[1] : next_point, ms=next_point_ms)
                    if show_combined_plot
                        push!(acq_plts, plt_acq)
                    elseif show_acquisition
                        plt = plot(plt_gp, plt_acq, size=(750,300), margin=3Plots.mm)
                        try
                            display(plt)
                        catch err
                            @warn err
                        end
                        if save_plots
                            plt_filename = joinpath(plots_dir, "plot-$(m)-$(a)")
                            savefig_dense(plt, plt_filename)
                        end
                    end
                end

                t += 1
            end


            if show_combined_plot
                acq_plts[1] = plot(acq_plts[1], title="operational refinement", titlefontsize=10)
                acq_plts[2] = plot(acq_plts[2], title="boundary refinement", titlefontsize=10)
                acq_plts[3] = plot(acq_plts[3], title="uncertainty exploration", titlefontsize=10)
                plt_surrogate_models = plot_combined(gp, models; surrogate=true, show_data=true, title="surrogate")
                plt_acquisitions = plot(acq_plts..., layout=(1,3))
                plt = plot(plt_surrogate_models, plt_acquisitions, layout=@layout([a{0.8h}; b]), size=(750, 650))
                try
                    display(plt)
                catch err
                    @warn err
                end
                acq_plts = []
                if save_plots
                    plt_filename = joinpath(plots_dir, "plot-combined-$m")
                    savefig_dense(plt, plt_filename)
                end
            end

            Y′ = System.evaluate(sparams, inputs; subdir=m, kwargs...)
            Y = vcat(Y, Y′)

            gp = gp_fit(X, Y; gp_args...)

            if show_plots && !show_acquisition
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

        @info "p(fail) estimate = $(p_estimate(gp, models))"

        return gp
    catch err
        alert("Error in `iteratively_sample`!")
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
