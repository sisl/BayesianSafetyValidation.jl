default(fontfamily="Computer Modern", framestyle=:box, palette=palette(:darkrainbow))

COLOR_FAIL = cgrad([:green, :white, :red])


"""
Get ranges of the model used by the predicted output `y` of the GP.
"""
function get_model_ranges(models, m=[200,200])
    model_1_range = range(models[1].range[1], models[1].range[end], length=m[1])
    model_2_range = range(models[2].range[1], models[2].range[end], length=m[2])
    return [model_1_range, model_2_range]
end


"""
Plot actual data points ran through the system.
Green indicates non-failure, red indicates failure.
"""
function plot_data!(X, Y)
    for i in eachindex(Y)
        k = X[:, i]
        v = Y[i]
        color = v ? :red : :green
        scatter!([k[1]], [k[2]], c=color, ms=4, marker=:square, label=false)
    end
    return plot!()
end


"""
Plot prediction of the GP as a soft decision boundary between [0,1].
"""
function plot_soft_boundary(gp, models; show_data=true, clamping=true)
    y = gp_output(gp, models; clamping)
    model_ranges = get_model_ranges(models, size(y))
    contourf(model_ranges[1], model_ranges[2], y, c=COLOR_FAIL, lc=:black, lw=0.2, clims=(0,1))
    if show_data
        plot_data!(gp.x, gp.y)
    end
    return plot!(xlabel=models[1].name, ylabel=models[2].name, title="probabilistic failure boundary", size=(450,400))
end


"""
Plot prediction of the GP as a hard decision boundary of either [0,1] given threshold of 0.5
"""
function plot_hard_boundary(gp, models; show_data=true, clamping=true)
    y = gp_output(gp, models; clamping)
    model_ranges = get_model_ranges(models, size(y))
    contourf(model_ranges[1], model_ranges[2], y .> 0.5, c=COLOR_FAIL, lc=:white)
    if show_data
        plot_data!(gp.x, gp.y)
    end
    return plot!(xlabel=models[1].name, ylabel=models[2].name, title="hard failure boundary", size=(450,400))
end


"""
Plot true function `f`
"""
function plot_truth(sparams, models; m=[200,200], overlay=false)
    f = (x,y)->System.evaluate(sparams, [[x,y]])[1]
    model_ranges = get_model_ranges(models, m)
    @suppress contourf(model_ranges[1], model_ranges[2], f, c=COLOR_FAIL, lc=:white)
    if overlay
        p(x,y) = pdf(models[1].distribution, x) * pdf(models[2].distribution, y)
        @suppress contour!(model_ranges[1], model_ranges[2], p, c=:white, levels=100)
    end
    return plot!(xlabel=models[1].name, ylabel=models[2].name, title="truth", size=(450,400))
end


"""
Plot true function, include data (when passing in `gp`).
"""
function plot_truth(gp, sparams, models; m=[200,200])
    plot_truth(sparams, models; m)
    plot_data!(gp.x, gp.y)
end


"""
Plot prediction of the GP as a hard decision boundary of either [0,1] given threshold of 0.5
"""
function plot_acquisition(gp, models; m=fill(200, length(models)), acq, acq_explore=nothing, return_point=false, show_point=true, given_next_point=nothing, as_pdf=false, clamping=false, ms=5)
    model_ranges = get_model_ranges(models, m)
    boundary_gp = gp_output(gp, models; m, f=acq, clamping)
    if !isnothing(acq_explore)
        explore_gp = gp_output(gp, models; m, f=acq_explore, clamping=false)
        explore_gp = normalize01(explore_gp)
        boundary_gp = normalize01(boundary_gp)
        boundary_gp = explore_gp + boundary_gp
        if all(isnan.(boundary_gp))
            boundary_gp = ones(size(boundary_gp))
        end
    end
    if as_pdf
        boundary_gp = normalize01(boundary_gp)
        boundary_gp = normalize(boundary_gp, 1)
    end
    if all(isnan.(boundary_gp))
        @warn "All NaNs in acquisition function, setting to all zeros."
        boundary_gp = size(zeros(boundary_gp))
    end
    contourf(model_ranges[1], model_ranges[2], boundary_gp, c=:viridis, lc=:black, lw=0.1) # , clims=(0, 0.5))
    if show_point
        if isnothing(given_next_point)
            # get max from acquisition function to show as next point
            next_point = get_next_point(gp, models; acq)
        else
            next_point = given_next_point
        end
        scatter!([next_point[1]], [next_point[2]], label=false, ms=ms, mc=:red)
    end
    plt = plot!(title="acquisition function", size=(450,400))
    if return_point && show_point
        return (plt, next_point)
    else
        return plt
    end
end


"""
Plot operational model distribution (density).
"""
function plot_model(model, m=500; rotated=false, left=false, label=false, fill=true, alpha=0.3)
    if rotated
        model_range = range(model.range[1], model.range[end], m)
        if left
            plt = plot([pdf(model.distribution, x) for x in model_range], model_range, fill=fill, fillalpha=alpha, label=label, xaxis=:flip, xrotation=90)
        else
            plt = plot([pdf(model.distribution, x) for x in model_range], model_range, fill=fill, fillalpha=alpha, label=label)
        end
    else
        plt = plot(range(model.range[1], model.range[end], m), x->pdf(model.distribution, x), fill=fill, fillalpha=alpha, label=label)
    end
    plot!(xtickfontsize=8, ytickfontsize=8)
    return plt
end


"""
Plot function with model distributions above and to the right.
"""
function plot_combined(gp, models; surrogate=false, soft=true, show_q=false, acq=nothing, show_point=true, title=nothing, mlf=false, label1=models[1].name, label2=models[2].name, show_data=true)
    if surrogate
        if soft
            plt_main = plot_soft_boundary(gp, models; show_data)
            if isnothing(title)
                title!("soft boundary")
            else
                title!(title)
            end
        else
            plt_main = plot_hard_boundary(gp, models; show_data)
            if isnothing(title)
                title!("hard boundary")
            else
                title!(title)
            end
        end
    elseif mlf
        plt_main, next_point = plot_most_likely_failure(gp, models; return_failure=true)
        title!("hard boundary")
    else
        if isnothing(acq)
            error("Please assign keywork `acq`")
        end
        if show_point
            plt_main, next_point = plot_acquisition(gp, models; acq, return_point=true)
        else
            plt_main = plot_acquisition(gp, models; acq, return_point=false, show_point=false)
        end
        xlabel!(label1)
        ylabel!(label2)
        if isnothing(title)
            title!("acquisition")
        else
            title!(title)
        end
    end

    xl = xlims()
    yl = ylims()
    plt_model1 = plot_model(models[1]; label=label1, fill=true)
    show_q && plot_q_proposal(gp, models, 1; show_p=false)
    xlims!(xl)

    if !surrogate && show_point
        model1_x = next_point[1]
        model1_p = pdf(models[1].distribution, model1_x)
        scatter!([model1_x], [model1_p], label=false, c=:red, ms=3)
        plot!([model1_x, model1_x], [0, model1_p], label=false, c=:red)
    end

    # Rotated.
    plt_model2 = plot_model(models[2]; label=label2, fill=true, rotated=true, left=true)
    show_q && plot_q_proposal(gp, models, 2; show_p=false, rotated=true)
    ylims!(yl)

    if !surrogate && show_point
        model2_x = next_point[2]
        model2_p = pdf(models[2].distribution, model2_x)
        scatter!([model2_p], [model2_x], label=false, c=:red, ms=3)
        plot!([0, model2_p], [model2_x, model2_x], label=false, c=:red)
    end

    plt_cbar = contourf([0], [0], (x,y)->0,
        clims=(0,1), levels=11, c=COLOR_FAIL, axis=false, tick=nothing, label=false)

    plt_main = plot(plt_main, cbar=false)

    return plot(plt_model1, plt_model2, plt_main, plt_cbar,
		        layout=@layout([_ a{0.25h} _; b{0.18w} c{0.45w} d{0.1w}]))
end


plot_acquisition_combined(gp, models; λ, δ, kwargs...) = plot_combined(gp, models; acq=(gp,x)->boundary_acquisition(gp, x, models; λ, δ), kwargs...)


"""
Plot target distribution (p, i.e., operational model) and q-proposal distribution fit using kernel density esimation (KDE).
"""
function plot_q_proposal(gp, models, model_i; show_p=true, rotated=false, failures_only=true)
    if show_p
        plot_model(models[model_i]; label="p")
    end
    if failures_only
        X = falsification(gp)[model_i, :]
    else
        X = gp.x[model_i, :]
    end
    U = kde(X)
    xr = get_model_ranges(models)[model_i]
    if rotated
        plot!([pdf(U, x) for x in xr], xr, label="q")
        scatter!(zeros(length(X)), X, c=:red, label="data")
    else
        plot!(xr, x->pdf(U, x), c=:red, label="q")
        scatter!(X, zeros(length(X)), c=:red, label="data")
    end
end


"""
Plot most-likely recorded failure.
"""
function plot_most_likely_failure(gp, models; return_failure=false)
    plot_hard_boundary(gp, models; show_data=false)
    x = most_likely_failure(gp, models)
    plt = scatter!([x[1]], [x[2]], c=:gold, lc=:white, ms=9, marker=:star5, label="most-likely failure")
    if return_failure
        return plt, x
    else
        return plt
    end
end


"""
Save PNG plot with high pixel density.
"""
function savefig_dense(plt, filename; density=600)
    savefig(plt, "$filename.svg")
    run(`convert -density $density $filename.svg $filename.png`)
    run(`rm $filename.svg`)
    return nothing
end
