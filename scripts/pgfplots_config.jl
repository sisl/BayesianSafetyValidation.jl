begin
	using Plots
	Plots.reset_defaults()
	pgfplotsx()
	default(
		thickness_scaling=1/2,
		titlefont=40,
		legendfontsize=28,
		guidefontsize=28,
		tickfontsize=28,
		colorbartickfontsizes=28,
		framestyle=:box,
		grid=true,
		linewidth=2,
		markersize=8,
		legend=:topright,
		size=(400,350),
	)
end
