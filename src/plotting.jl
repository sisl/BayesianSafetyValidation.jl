Plots.default(fontfamily="Times", framestyle=:box, palette=palette(:darkrainbow))

global DEFAULT_COLOR_FAIL = cgrad(["#007662", :white, "#8c1515"]) # cgrad([:green, :white, :red])
global COLOR_FAIL = DEFAULT_COLOR_FAIL

function set_colors!(colors)
    global COLOR_FAIL = cgrad(colors)
end

function reset_colors!()
    global COLOR_FAIL = DEFAULT_COLOR_FAIL
end


"""
Get ranges of the model used by the predicted output `y` of the GP.
"""
function get_model_ranges(models, m=fill(200, length(models)))
    return [range(model.range[1], model.range[end], length=l) for (model, l) in zip(models, m)]
end


get_lower_bounds(models) = [model.range[1] for model in models]
get_upper_bounds(models) = [model.range[2] for model in models]


"""
Plot actual data points ran through the system.
Green indicates non-failure, red indicates failure.
"""
function plot_data!(X, Y; ms=4, plot_inds=[1, 2], use_circles=false)
    if use_circles
        circle(x, y, r) = Plots.Shape(r*sind.(0:10:360) .+ x, r*cosd.(0:10:360) .+ y)
    end

    for i in eachindex(Y)
        k = X[:, i]
        v = Y[i]
        color = v >= 0 ? COLOR_FAIL[end] : COLOR_FAIL[1]
        if use_circles
            circles = circle.([k[plot_inds[1]]], [k[plot_inds[2]]]..., (ms,))
            plot!(circles, ratio=1, fc=:transparent, lc=:lightgray, lw=1, label=false)
        else
            scatter!([k[plot_inds[1]]], [k[plot_inds[2]]], c=color, ms=ms, marker=:square, label=false)
        end
    end
    return plot!()
end


"""
Plot prediction of the GP as a soft decision boundary between [0,1].
"""
function plot_soft_boundary(gp, models; inds=[:, :, fill(1, length(models) - 2)...], num_steps=200, m=fill(num_steps, length(models)), show_data=true, use_circles=false, overlay=false, overlay_levels=50, ms=4, lw=0, tight=false) # lw=0.2
    y = gp_output(gp, models; m)[inds...]
    s1, s2 = findall(isequal(:), inds)
    full_models = [models[s1], models[s2]]
    model_ranges = get_model_ranges(full_models, size(y))
    contourf(model_ranges[1], model_ranges[2], y'; c=COLOR_FAIL, lc=:black, lw=lw, clims=(0,1))
    if overlay
        p(x,y) = pdf(full_models[1].distribution, x) * pdf(full_models[2].distribution, y)
        @suppress contour!(model_ranges[1], model_ranges[2], p, c=cgrad([:gray, :black, :black, :white], 10, categorical=true, scale=:exp, rev=false), levels=overlay_levels)
    end
    if show_data
        plot_data!(gp.x, gp.y; ms, plot_inds=[s1, s2], use_circles)
    end
    if tight
        return plot!(cbar=false, ticks=false, xlabel="", ylabel="", title="", size=(400,400))
    else
        return plot!(xlabel="$(full_models[1].name)", ylabel="$(full_models[2].name)", title="probabilistic failure boundary", size=(450,400))
    end
end


"""
Plot prediction of the GP as a hard decision boundary of either [0,1] given threshold of 0.5
"""
function plot_hard_boundary(gp, models; inds=[:, :, fill(1, length(models) - 2)...], num_steps=200, m=fill(num_steps, length(models)), show_data=true, use_circles=false, overlay=false, overlay_levels=50, ms=4, tight=false, lw=1)
    y = gp_output(gp, models; m)[inds...]
    s1, s2 = findall(isequal(:), inds)
    full_models = [models[s1], models[s2]]
    model_ranges = get_model_ranges(full_models, size(y))
    contourf(model_ranges[1], model_ranges[2], y' .>= 0.5, c=COLOR_FAIL, lc=:white, lw=lw)
    if overlay
        p(x,y) = pdf(full_models[1].distribution, x) * pdf(full_models[2].distribution, y)
        @suppress contour!(model_ranges[1], model_ranges[2], p, c=cgrad([:gray, :black, :black, :white], 10, categorical=true, scale=:exp, rev=false), levels=overlay_levels)
    end
    if show_data
        plot_data!(gp.x, gp.y; ms, plot_inds=[s1, s2], use_circles)
    end
    if tight
        return plot!(cbar=false, ticks=false, xlabel="", ylabel="", title="", size=(400,400))
    else
        return plot!(xlabel="$(full_models[1].name)", ylabel="$(full_models[2].name)", title="hard failure boundary", size=(450,400))
    end
end


"""
Plot true function `f`
"""
function plot_truth(sparams, models; inds=[:, :, fill(1, length(models) - 2)...], num_steps=200, m=fill(num_steps, length(models)), overlay=false, overlay_levels=50, use_heatmap=false, tight=false, hard=false, lw=1, edges=false)
    model_ranges = get_model_ranges(models, m)
    s1, s2 = findall(isequal(:), inds)
    function f(x...)
        fx = @suppress System.evaluate(sparams, [x])[1]
        return hard ? fx >= 0.5 : fx
    end
    params = Any[collect(r[d]) for (r, d) in zip(model_ranges, inds)]
    params[s1] = params[s1]'

    y = f.(params...)

    plot_f = use_heatmap ? heatmap : contourf
    plot_f(model_ranges[s1], model_ranges[s2], y, c=COLOR_FAIL, lc=:white, clims=(0,1), lw=lw)
    if overlay
        p(x,y) = pdf(models[s1].distribution, x) * pdf(models[s2].distribution, y)
        @suppress contour!(model_ranges[s1], model_ranges[s2], p, c=cgrad([:gray, :black, :black, :white], 10, categorical=true, scale=:exp, rev=false), levels=overlay_levels)
    end
    if edges
        scatter!([model_ranges[1][1], model_ranges[1][end]], [model_ranges[1][1], model_ranges[1][end]], label=false, ms=0)
    end
    if tight
        return plot!(cbar=false, ticks=false, xlabel="", ylabel="", title="", size=(400,400))
    else
        return plot!(xlabel="\$$(models[s1].name)\$", ylabel="\$$(models[s2].name)\$", title="truth", size=(450,400))
    end
end


"""
Plot true function, include data (when passing in `gp`).
"""
function plot_truth(gp, sparams, models; inds=[:, :, fill(1, length(models) - 2)...], num_steps=200, m=fill(num_steps, length(models)), ms=4)
    plot_truth(sparams, models; inds, m)
    plot_inds = [i for (i, d) in enumerate(inds) if d == :]
    plot_data!(gp.x, gp.y; plot_inds, ms)
end


"""
Plot prediction of the GP as a hard decision boundary of either [0,1] given threshold of 0.5
"""
function plot_acquisition(y, F̂, P, models; inds=[:, :, fill(1, length(models) - 2)...], acq, zero_white=false, return_point=false, show_point=true, given_next_point=nothing, as_pdf=false, tight=false, ms=5, lw=0) # lw=0.1
    full_models = [m for (d, m) in zip(inds, models) if d == :]
    model_ranges = get_model_ranges(full_models, size(y))
    acq_output = map(acq, F̂, P)
    if as_pdf
        acq_output = normalize01(acq_output)
        acq_output = normalize(acq_output, 1)
    end
    if all(isnan.(acq_output))
        @warn "All NaNs in acquisition function, setting to all zeros."
        acq_output = zeros(size(acq_output))
    end
    if zero_white
        acq_output[acq_output .== 0] .= NaN
    end
    contourf(model_ranges[1], model_ranges[2], acq_output[inds...]', c=:viridis, lc=:black, lw=lw, fill=!zero_white)
    if show_point
        if isnothing(given_next_point)
            # get max from acquisition function to show as next point
            next_point = get_next_point(y, F̂, P, models; acq)
        else
            next_point = given_next_point
        end
        inds = [i for (i, ind) in enumerate(inds) if ind == :]
        scatter!([next_point[inds[1]]], [next_point[inds[2]]], label=false, ms=ms, mc=:red)
    end
    plt = plot!(title="acquisition function", size=(450,400))
    if tight
        plot!(colorbar=false, ticks=false, xlabel="", ylabel="")
    end
    if return_point && show_point
        return (plt, next_point)
    else
        return plt
    end
end


"""
Plot operational model distribution (density).
"""
function plot_model(model, m=500; rotated=false, left=false, label=false, fill=true, alpha=0.3, tight=false)
    model_range = range(model.range[1], model.range[end], m)
    limscale = isa(model.distribution, Uniform) ? 1.4 : 1.2
    Y = [pdf(model.distribution, x) for x in model_range]
    X = collect(model_range)
    # hack to get fill region all the way to zero
    Y = [0, Y..., 0]
    X = [X[1], X..., X[end]]
    if rotated
        if left
            plt = plot(Y, X, fill=fill, fillalpha=alpha, label=label, legendfont=(12, "Computer Modern"), xaxis=:flip, xrotation=90, c=:gray)
        else
            plt = plot(Y, X, fill=fill, fillalpha=alpha, label=label, legendfont=(12, "Computer Modern"), c=:gray)
        end
        if tight
            plot!(xticks=false)
        end
        xlims!(0, maximum(Y)*limscale)
    else
        plt = plot(X, Y, fill=fill, fillalpha=alpha, label=label, legendfont=(12, "Computer Modern"), c=:gray)
        if tight
            plot!(yticks=false)
        end
        ylims!(0, maximum(Y)*limscale)
    end
    if tight
        plot!(xtickfontsize=12, ytickfontsize=12, legendfontsize=10)
    else
        plot!(xtickfontsize=8, ytickfontsize=8)
    end
    return plt
end


"""
Plot function with model distributions above and to the right.
"""
function plot_combined(gp, models, sparams; inds=[:, :, fill(1, length(models) - 2)...], num_steps=200, m=fill(num_steps, length(models)), truth=false, surrogate=true, soft=true, show_q=false, acq=nothing, show_point=true, title=nothing, mlf=false, latex_labels=false, show_data=true, tight=false, use_heatmap=false, overlay=false, overlay_levels=50, acq_plts=nothing, hide_model=false, hide_ranges=false, titlefontsize=12, add_phantom_point=false, include_surrogate=false)
    s1, s2 = findall(isequal(:), inds)
    label1 = latex_labels ? " \$p($(models[s1].name))\$" : "\$p(\$$(models[s1].name)\$)\$"
    label2 = latex_labels ? " \$p($(models[s2].name))\$" : "\$p(\$$(models[s2].name)\$)\$"

    ms = tight ? 3 : 4

    if truth
        plt_main = plot_truth(sparams, models; inds, m, overlay, overlay_levels, use_heatmap)
        if !isnothing(title)
            title!(title)
        end
    elseif surrogate
        if soft
            plt_main = plot_soft_boundary(gp, models; inds, m, show_data, ms, overlay, overlay_levels, tight)
            if isnothing(title)
                title!("soft boundary")
            else
                title!(title)
            end
        else
            plt_main = plot_hard_boundary(gp, models; inds, m, show_data, ms, overlay, overlay_levels, tight)
            if isnothing(title)
                title!("hard boundary")
            else
                title!(title)
            end
        end
    elseif mlf
        plt_main, next_point = plot_most_likely_failure(gp, models; m, inds, return_failure=true)
        title!("hard boundary")
    else
        if isnothing(acq)
            error("Please assign keywork `acq`")
        end
        F̂ = gp_output(gp, models; f=predict_f_vec, m)
        P = p_output(models; m)
        y = F̂
        if show_point
            plt_main, next_point = plot_acquisition(y, F̂, P, models; inds, acq, return_point=true)
        else
            plt_main = plot_acquisition(y, F̂, P, models; inds, acq, return_point=false, show_point=false)
        end
        xlabel!(label1, legendfont=(12, "Computer Modern"))
        ylabel!(label2, legendfont=(12, "Computer Modern"))
        if isnothing(title)
            title!("acquisition")
        else
            title!(title)
        end
    end

    if tight
        plt_main = plot!(plt_main, ticks=false, xlabel="", ylabel="", title=hide_model ? title : "", titlefont=(titlefontsize, "Times"))

        if add_phantom_point
            # Add phantom scatter to mimic acquisition plot's xy boundaries
            model_ranges = get_model_ranges(models)
            left_corner = [model_ranges[s1][1], model_ranges[s2][1]]
            scatter!(plt_main, [left_corner[1]], [left_corner[2]], label=false, ms=ms, mc=:white, alpha=0)
        end
    end

    xl = xlims()
    yl = ylims()
    if hide_model
        # Mimic model plots to preserve spacing
        plt_model1 = plot([0], [0], axis=false, label=false, grid=false, tickfont=:white, yticks=false, xtickfontsize=12, ytickfontsize=12, legendfontsize=10)
    else
        plt_model1 = plot_model(models[s1]; label=label1, fill=true, tight)
        if hide_ranges
            plt_model1 = plot!(xticks=false)
        end
    end
    show_q && plot_q_proposal(gp, models, 1; show_p=false)
    xlims!(xl)

    if !surrogate && !truth && show_point
        model1_x = next_point[s1]
        model1_p = pdf(models[s1].distribution, model1_x)
        scatter!([model1_x], [model1_p], label=false, c=:red, ms=ms)
        plot!([model1_x, model1_x], [0, model1_p], label=false, c=:red)
    end

    # Rotated.
    if hide_model
        # Mimic model plots to preserve spacing
        plt_model2 = plot([0], [0], axis=false, label=false, grid=false, tickfont=:white, xticks=false, xtickfontsize=12, ytickfontsize=12, legendfontsize=10)
    else
        plt_model2 = plot_model(models[s2]; label=label2, fill=true, rotated=true, left=true, tight)
        if hide_ranges
            plt_model2 = plot!(yticks=false)
        end
    end
    show_q && plot_q_proposal(gp, models, 2; show_p=false, rotated=true)
    ylims!(yl)

    if !surrogate && !truth && show_point
        model2_x = next_point[s2]
        model2_p = pdf(models[s2].distribution, model2_x)
        scatter!([model2_p], [model2_x], label=false, c=:red, ms=3)
        plot!([0, model2_p], [model2_x, model2_x], label=false, c=:red)
    end

    if !tight
        plt_cbar = contourf([0], [0], (x,y)->0,
            clims=(0,1), levels=tight ? 2 : 10, c=COLOR_FAIL, axis=false, tick=nothing, label=false)
        # plt_cbar = plot!(plt_cbar, tickfontsize=14)
    end

    plt_main = plot(plt_main, cbar=false)

    if tight
        if isnothing(acq_plts)
            if include_surrogate
                plt_main = plot!(plt_main, title="truth")
                plt_gp = plot_hard_boundary(gp, models; inds, m, show_data, ms)
                plot!(plt_gp, ticks=false, title="surrogate", ylabel="", xlabel="", titlefont=(titlefontsize, "Times"), colorbar=false)
                lo = @layout([_ a{0.17647058823529413h}; b c{0.7w, 0.4117647058823529h}; _ d{0.7w}])
                return plot(plt_model1, plt_model2, plt_main, plt_gp, layout=lo, size=(400,400*0.3 + 400*0.7*2))
            else
                lo = @layout([_ a{0.30h}; b c{0.7w, 0.7h}])
                return plot(plt_model1, plt_model2, plt_main, layout=lo, size=(400,400))
            end
        else
            ## Note: how the widths were calculated
            # ratio1, ratio2 = normalize([0.3, 0.7*4], 1)
            # ratio2 = ratio2 / 4
            lo = @layout([_ a{0.30h} _ _ _; b{0.09677419354838711w} c{0.22580645161290325w} d{0.22580645161290325w} e{0.22580645161290325w} f{0.22580645161290325w}])
            return plot(plt_model1, plt_model2, plt_main, acq_plts..., layout=lo, size=(400*0.3 + 400*0.7*4,400))
        end
    else
        lo = @layout([_ a{0.25h} _; b{0.18w} c{0.45w} d{0.1w}])
        return plot(plt_model1, plt_model2, plt_main, plt_cbar, layout=lo)
    end
end


function plot_surrogate_truth_combined(gp, models, sparams; inds=[:, :, fill(1, length(models) - 2)...], num_steps=200, m=fill(num_steps, length(models)), latex_labels=true, show_data=false, overlay=false, tight=true, use_heatmap=false, hide_model=true, hide_ranges=false, titlefontsize=12, add_phantom_point=true)
    s1, s2 = findall(isequal(:), inds)
    label1 = latex_labels ? " \$p($(models[s1].name))\$" : "\$p(\$$(models[s1].name)\$)\$"
    label2 = latex_labels ? " \$p($(models[s2].name))\$" : "\$p(\$$(models[s2].name)\$)\$"
    ms = tight ? 3 : 4

    function phantom_point()
        if add_phantom_point
            # Add phantom scatter to mimic acquisition plot's xy boundaries
            model_ranges = get_model_ranges(models)
            left_corner = [model_ranges[s1][1], model_ranges[s2][1]]
            scatter!([left_corner[1]], [left_corner[2]], label=false, ms=ms, mc=:white, alpha=0)
        end
    end

    plt_truth_soft = plot_truth(sparams, models; inds, m, overlay, tight, use_heatmap, hard=false, lw=0)
    phantom_point()
    title!("truth (soft)", titlefont=(titlefontsize, "Times"))

    plt_truth_hard = plot_truth(sparams, models; inds, m, overlay, tight, use_heatmap, hard=true)
    phantom_point()
    title!("truth (hard)", titlefont=(titlefontsize, "Times"))

    plt_surrogate_soft = plot_soft_boundary(gp, models; inds, m, show_data, ms, overlay, tight)
    phantom_point()
    title!("surrogate (soft)", titlefont=(titlefontsize, "Times"))

    plt_surrogate_hard = plot_hard_boundary(gp, models; inds, m, show_data, ms, overlay, tight)
    phantom_point()
    title!("surrogate (hard)", titlefont=(titlefontsize, "Times"))

    xl = xlims()
    yl = ylims()
    if hide_model
        # Mimic model plots to preserve spacing
        plt_model1 = plot([0], [0], axis=false, label=false, grid=false, tickfont=:white, yticks=false, xtickfontsize=12, ytickfontsize=12, legendfontsize=10)
    else
        plt_model1 = plot_model(models[s1]; label=label1, fill=true, tight)
        if hide_ranges
            plt_model1 = plot!(xticks=false)
        end
    end
    xlims!(xl)

    # Rotated.
    if hide_model
        # Mimic model plots to preserve spacing
        plt_model2 = plot([0], [0], axis=false, label=false, grid=false, tickfont=:white, xticks=false, xtickfontsize=12, ytickfontsize=12, legendfontsize=10)
    else
        plt_model2 = plot_model(models[s2]; label=label2, fill=true, rotated=true, left=true, tight)
        if hide_ranges
            plt_model2 = plot!(yticks=false)
        end
    end
    ylims!(yl)

    # if !tight
    # plt_cbar = contourf!([0], [0], (x,y)->0,
    #     clims=(0,1), levels=10, c=COLOR_FAIL, axis=false, tick=nothing, label=false)
        # plt_cbar = plot!(plt_cbar, tickfontsize=14)
    # end


    if tight
        ## Note: how the widths were calculated
        # ratio1, ratio2 = normalize([0.3, 0.7*4], 1)
        # ratio2 = ratio2 / 4
        lo = @layout([_ a{0.30h} _ _ _; b{0.09677419354838711w} c{0.22580645161290325w} d{0.22580645161290325w} e{0.22580645161290325w} f{0.22580645161290325w}])
        plot(plt_model1, plt_model2, plt_surrogate_soft, plt_truth_soft, plt_surrogate_hard, plt_truth_hard, layout=lo, size=(400*0.3 + 400*0.7*4,400))
        if hide_model
            contourf!([0], [0], (x,y)->0,
                clims=(0,1), levels=10, c=COLOR_FAIL,
                ytick=true, # Hack, which is trimmed anyhow.
                bg_inside=nothing,
                subplot=2,
                axis=false,
                label=false,
                grid=false,
                tickfont=:black, # Important. We set to :white for hidden plt_model2 above.
                ytickfontsize=12,
            )
        end
        return plot!()
    else
        lo = @layout([_ a{0.25h} _; b{0.18w} c{0.45w} d{0.1w}])
        return plot(plt_model1, plt_model2, plt_main, plt_cbar, layout=lo)
    end
end


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
function plot_most_likely_failure(gp, models; inds=[:, :, fill(1, length(models) - 2)...], num_steps=200, m=fill(num_steps, length(models)), return_failure=false, hard=false, lw=1)
    hard ? plot_hard_boundary(gp, models; inds, m, show_data=false, lw) : plot_soft_boundary(gp, models; inds, m, show_data=false, lw)
    x = most_likely_failure(gp, models)
    s1, s2 = findall(isequal(:), inds)
    c = :white
    plt = plot!([x[s1]-0.6, x[s1]-0.04], [x[s2], x[s2]], c=c, arrow=(:closed, 2.0), label=false)
    # px = round(pdf(models, x), digits=3)
    # mlf_text = "most-likely failure\n\$p(\\mathbf{x}) \\approx $px\$"
    mlf_text = "most-likely\nfailure"
    annotate!(x[s1]-1.3, x[s2], text(mlf_text, c, :center, 14, "Computer Modern"))
    # plt = scatter!([x[1]], [x[2]], c=:gold, lc=:white, ms=9, marker=:star5, alpha=0.5, label="most-likely failure")
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


function plot_distribution_of_failures(gp, models)
    histogram([log(pdf(models, x)) for x in eachcol(gp.x)],
        c="#8C1515",
        lc="#F4F4F4",
        label=false,
        margin=5Plots.mm,
        size=(700,300),
        xlabel="log-likelihood of observed failure",
        ylabel="count",
    )
end


function plot1d(gp, models;
                num_steps=200,
                num_steps_obs=100,
                scale=1.5,
                show_obs=true,
                show_surrogate=true,
                show_surrogate_uncertainty=false,
                show_surrogate_pred=true)

    Xr = get_model_ranges(models, num_steps)[1]
    Xrobs = get_model_ranges(models, min(num_steps, num_steps_obs))[1]
    plot(Xrobs, x->pdf(models[1], x); lw=0.5, ls=:dash, c=:gray, label="operational model") # \$p(x)\$")

    X = gp.x'
    Y = inverse.(gp.y)

    # Surrogate
    if show_surrogate
        ŷ = map(x->f_gp(gp, x), Xr)
        if show_surrogate_uncertainty
            ribbon = map(x->σ_gp(gp, x), Xr)
        else
            ribbon = nothing
        end
        gp_color = :black # :steelblue
        plot!(Xr, ŷ, ribbon=ribbon, c=gp_color, lw=0.5, label="surrogate") # \$\\hat{f}(x)\$")
    end

    if show_surrogate_pred
        ŷ_fail = map(x->g_gp(gp, x), Xr)
        # plot!(Xr, ŷ_fail, c=:black, lw=0.5, ls=:dot, label="surrogate hard prediction") # \$\\hat{g}(x)\$")
        pred_fails = ŷ_fail .== 1
        pred_succs = ŷ_fail .== 0
        pred_mark = :vline
        pred_ms = 1
        pred_msw = 0
        pred_fail_color = :red
        pred_succ_color = :limegreen
        # for i in [0.0475, 0.05, 0.0525]
        for i in [0.05]
            scatter!(Xr[pred_fails], (1+i)*ones(sum(pred_fails)), label=false, c=pred_fail_color, msc=pred_fail_color, msw=pred_msw, ms=pred_ms, mark=pred_mark)
            scatter!(Xr[pred_succs], (1+i)*ones(sum(pred_succs)), label=false, c=pred_succ_color, msc=pred_succ_color, msw=pred_msw, ms=pred_ms, mark=pred_mark)
        end
    end


    # Observations
    if show_obs && !isempty(X)
        obs_fail = Y .== 1
        obs_succ = Y .== 0
        scatter!(X[obs_fail], Y[obs_fail], c=:red, ms=1.5, msw=0.5, mark=:square, alpha=0.5, label=false) #, label="observed failures") # \$\\mathbf{Y}\$")
        scatter!(X[obs_succ], Y[obs_succ], c=:green, ms=1.5, msw=0.5, mark=:square, alpha=0.5, label=false) #, label="observed successes") # \$\\mathbf{Y}\$")
    end

    xl = (1.005first(Xr), 1.005last(Xr))
	yl = (-0.05, 1.1)
    yticks = [0, 1] # 0:0.25:1

    xlabel!("initial state")
    ylabel!("probability")

    plot!(xlims=xl, ylims=yl, legend=:left, size=(600, 300) ./ scale, yticks=yticks)
end
