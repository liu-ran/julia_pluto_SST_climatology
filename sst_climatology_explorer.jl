### A Pluto.jl notebook ###
# v0.20.8

using Markdown
using InteractiveUtils

# ╔═╡ 7d6a69f6-6b7d-4c4d-a11b-000000000001
begin
	using NCDatasets
	using Dates
	using DataFrames
	using Statistics
	using GLM
	using StatsModels
	using Plots
	using LaTeXStrings
	using PlutoUI
end

# ╔═╡ 7d6a69f6-6b7d-4c4d-a11b-000000000002
md"""
# SST climatology / anomaly explorer

This Pluto notebook reorganizes your original Julia script into a Binder-friendly interactive page.

## What is interactive here?
- The **zoom figure** is controlled by a slider.
- You can drag the slider to change the **x-limits (date range)** of the zoom-in plots.
- The full-period figure stays fixed for reference.

## Binder / GitHub notes
- Put this notebook in a GitHub repository.
- Add the accompanying `Project.toml` file.
- Launch with Binder and open Pluto.

The notebook assumes your NetCDF file is placed in the same repository folder (or uploaded into the Binder session).
"""

# ╔═╡ 7d6a69f6-6b7d-4c4d-a11b-000000000003
begin
	# ---------------------------
	# User settings
	# ---------------------------
	datafile = "./ECCOv4r4_Nino3_point_SST_daily_19920101_20171231.nc"
	varname = "sst"

	# climatology reference period
	t_clim_start = Date(1992, 1, 1)
	t_clim_end   = Date(2016, 12, 31)

	# GLM harmonic order: 0,1,2,3
	glm_order = 2

	# default zoom period shown when notebook opens
	default_zoom_start = Date(2015, 11, 30)
	default_zoom_end   = Date(2016, 4, 1)

	# plotting defaults
	default(
		legend = :outerright,
		legendfontsize = 12,
		guidefontsize = 13,
		tickfontsize = 10,
		titlefontsize = 12,
		dpi = 130,
		size = (1100, 900)
	)
end

# ╔═╡ 7d6a69f6-6b7d-4c4d-a11b-000000000004
begin
	"""
	    daily_derivative(time, x)

	Compute numerical time derivative on a daily Date axis.
	- first point: forward difference
	- interior: centered difference
	- last point: backward difference
	Units follow x/day.
	"""
	function daily_derivative(time::Vector{Date}, x::AbstractVector{<:Real})
		n = length(x)
		@assert length(time) == n
		@assert n >= 3

		xx = Float64.(x)
		dxdt = similar(xx)

		dt1 = Float64(Dates.value(time[2] - time[1]))
		dxdt[1] = (xx[2] - xx[1]) / dt1

		for i in 2:n-1
			dt = Float64(Dates.value(time[i+1] - time[i-1]))
			dxdt[i] = (xx[i+1] - xx[i-1]) / dt
		end

		dtn = Float64(Dates.value(time[end] - time[end-1]))
		dxdt[end] = (xx[end] - xx[end-1]) / dtn

		return dxdt
	end

	"""
	    daily_climatology_from_reference(time_full, x_full, mask_ref; clim_name=:clim)

	Strict daily climatology based on mmdd grouping, including leap day.
	The climatology is computed only from the reference subset defined by `mask_ref`,
	then mapped back to the full time axis.
	"""
	function daily_climatology_from_reference(
		time_full::Vector{Date},
		x_full::Vector{Float64},
		mask_ref::AbstractVector{Bool};
		clim_name::Symbol = :clim
	)
		df_full = DataFrame(time = time_full, x = x_full)
		df_full.mmdd = Dates.format.(df_full.time, "mmdd")

		df_ref = df_full[mask_ref, :]

		clim = combine(
			groupby(df_ref, :mmdd),
			:x => (v -> mean(skipmissing(v))) => clim_name
		)

		df_out = leftjoin(df_full, clim, on = :mmdd)
		clim_on_full = Float64.(df_out[!, clim_name])

		return clim_on_full, df_out
	end

	"""
	    fit_glm_background_from_reference(y, t_days, mask_ref; order=2)

	Fit trend + annual/semiannual/triannual harmonics using only the reference period,
	then predict over the full record.
	"""
	function fit_glm_background_from_reference(
		y::Vector{Float64},
		t_days::Vector{Float64},
		mask_ref::AbstractVector{Bool};
		order::Int = 2
	)
		year_days = 365.2425

		c1 = cos.(2π .* t_days ./ year_days)
		s1 = sin.(2π .* t_days ./ year_days)
		c2 = cos.(4π .* t_days ./ year_days)
		s2 = sin.(4π .* t_days ./ year_days)
		c3 = cos.(6π .* t_days ./ year_days)
		s3 = sin.(6π .* t_days ./ year_days)

		df_full = DataFrame(
			y = y,
			t = t_days,
			c1 = c1, s1 = s1,
			c2 = c2, s2 = s2,
			c3 = c3, s3 = s3
		)

		df_ref = df_full[mask_ref, :]

		if order == 0
			mdl = lm(@formula(y ~ t), df_ref)
		elseif order == 1
			mdl = lm(@formula(y ~ t + c1 + s1), df_ref)
		elseif order == 2
			mdl = lm(@formula(y ~ t + c1 + s1 + c2 + s2), df_ref)
		elseif order == 3
			mdl = lm(@formula(y ~ t + c1 + s1 + c2 + s2 + c3 + s3), df_ref)
		else
			error("order must be 0, 1, 2, or 3")
		end

		bg_full = Float64.(coalesce.(predict(mdl, df_full), NaN))
		return bg_full, mdl
	end
end

# ╔═╡ 7d6a69f6-6b7d-4c4d-a11b-000000000005
begin
	# ---------------------------
	# Read data
	# ---------------------------
	@assert isfile(datafile) "NetCDF file not found: $(datafile). Put it next to this notebook or update `datafile`."

	f = NCDataset(datafile)
	sst  = Float64.(vec(f[varname][:]))
	time = Date.(f["time"][:])
	close(f)

	mask_clim = (time .>= t_clim_start) .& (time .<= t_clim_end)
	idx_clim = findall(mask_clim)
	i0 = first(idx_clim)
	i1 = last(idx_clim)
	t0_clim = time[i0]

	# time relative to climatology start
	t_rel_days = Float64.(Dates.value.(time .- t0_clim))
	Δt_clim_days = Float64(Dates.value(time[i1] - time[i0]))

	nothing
end

# ╔═╡ 7d6a69f6-6b7d-4c4d-a11b-000000000006
begin
	# ---------------------------
	# Backgrounds and anomalies
	# ---------------------------
	# fixed daily climatology anomaly: Θ'
	sst_bg_fixed, _ = daily_climatology_from_reference(
		time, sst, mask_clim; clim_name = :clim_fixed
	)
	sst_anom_fixed = sst .- sst_bg_fixed

	# detrended anomaly: \tilde{Θ}'
	mean_tendency_day = (sst[i1] - sst[i0]) / Δt_clim_days
	sst_detrended = sst .- mean_tendency_day .* t_rel_days
	sst_bg_det_core, _ = daily_climatology_from_reference(
		time, sst_detrended, mask_clim; clim_name = :clim_det
	)
	sst_anom_det = sst_detrended .- sst_bg_det_core

	# GLM anomaly: Θ'_GLM
	sst_bg_glm, mdl_glm = fit_glm_background_from_reference(
		sst, t_rel_days, mask_clim; order = glm_order
	)
	sst_anom_glm = sst .- sst_bg_glm

	nothing
end

# ╔═╡ 7d6a69f6-6b7d-4c4d-a11b-000000000007
begin
	# ---------------------------
	# Derivatives and true tendency anomalies
	# ---------------------------
	d_anom_fixed_dt = daily_derivative(time, sst_anom_fixed)
	d_anom_det_dt   = daily_derivative(time, sst_anom_det)
	d_anom_glm_dt   = daily_derivative(time, sst_anom_glm)

	dTdt = daily_derivative(time, sst)

	# fixed climatology truth: (∂tT)' = ∂tT - overline(∂tT)
	dTdt_bg_fixed, _ = daily_climatology_from_reference(
		time, dTdt, mask_clim; clim_name = :clim_dTdt_fixed
	)
	dTdt_anom_fixed_true = dTdt .- dTdt_bg_fixed

	# detrended truth
	dTdt_det_temp = dTdt .- mean_tendency_day
	dTdt_bg_det_core, _ = daily_climatology_from_reference(
		time, dTdt_det_temp, mask_clim; clim_name = :clim_dTdt_det
	)
	dTdt_anom_det_true = dTdt_det_temp .- dTdt_bg_det_core

	# GLM truth
	dTdt_bg_glm, mdl_glm_dTdt = fit_glm_background_from_reference(
		dTdt, t_rel_days, mask_clim; order = glm_order
	)
	dTdt_anom_glm_true = dTdt .- dTdt_bg_glm

	nothing
end

# ╔═╡ 7d6a69f6-6b7d-4c4d-a11b-000000000008
begin
	# ---------------------------
	# Diagnostics panel
	# ---------------------------
	md"""
	**Loaded** $(length(sst)) daily points  
	**Time range:** $(first(time)) to $(last(time))  
	**Climatology reference period:** $(t_clim_start) to $(t_clim_end)  
	**GLM order:** $(glm_order)
	"""
end

# ╔═╡ 7d6a69f6-6b7d-4c4d-a11b-000000000009
begin
	# ---------------------------
	# Full-period figure
	# ---------------------------
	p1 = plot(time, sst, lw = 0.8, label = L"\Theta", color = :black)
	plot!(p1, time, sst_bg_fixed,    lw = 1.4, label = L"\overline{\Theta}")
	plot!(p1, time, sst_bg_det_core, lw = 1.4, label = L"\overline{\tilde{\Theta}}")
	plot!(p1, time, sst_bg_glm,      lw = 1.4, label = L"\overline{\Theta}_{\mathrm{GLM}}")
	ylabel!(p1, L"\Theta\ (\mathrm{^\circ C})")
	title!(p1, "(a) Background / climatology comparison")

	p2 = plot(time, sst_anom_fixed, lw = 0.8, label = L"\Theta'")
	plot!(p2, time, sst_anom_det, lw = 0.8, label = L"\tilde{\Theta}'")
	plot!(p2, time, sst_anom_glm, lw = 0.8, label = L"\Theta'_{\mathrm{GLM}}")
	ylabel!(p2, L"\mathrm{Anomaly\ (^\circ C)}")
	title!(p2, "(b) Anomaly comparison")

	p3 = plot(time, d_anom_fixed_dt, lw = 0.7, label = L"\partial_t \Theta'")
	plot!(p3, time, d_anom_det_dt, lw = 0.7, label = L"\partial_t \tilde{\Theta}'")
	plot!(p3, time, d_anom_glm_dt, lw = 0.7, label = L"\partial_t \Theta'_{\mathrm{GLM}}")
	ylabel!(p3, L"\mathrm{^\circ C\ day^{-1}}")
	title!(p3, "(c) Derivative of anomaly")

	p4 = plot(time, dTdt_anom_fixed_true, lw = 0.7, label = L"(\partial_t T)'")
	plot!(p4, time, dTdt_anom_det_true, lw = 0.7, label = L"(\partial_t T)'_{\mathrm{det}}")
	plot!(p4, time, dTdt_anom_glm_true, lw = 0.7, label = L"(\partial_t T)'_{\mathrm{GLM}}")
	ylabel!(p4, L"\mathrm{^\circ C\ day^{-1}}")
	title!(p4, "(d) True tendency anomaly")

	plot(p1, p2, p3, p4, layout = (4,1), size = (1200, 920))
end

# ╔═╡ 7d6a69f6-6b7d-4c4d-a11b-000000000010
begin
	# ---------------------------
	# Interactive zoom selector
	# ---------------------------
	default_i_start = findfirst(>=(default_zoom_start), time)
	default_i_end = findlast(<=(default_zoom_end), time)

	if isnothing(default_i_start)
		default_i_start = 1
	end
	if isnothing(default_i_end)
		default_i_end = length(time)
	end

	@bind zoom_idx RangeSlider(1:length(time), default = (default_i_start, default_i_end), show_value = true)
end

# ╔═╡ 7d6a69f6-6b7d-4c4d-a11b-000000000011
begin
	# normalize slider output and protect against reversed ranges
	i_start, i_end = zoom_idx isa Tuple ? zoom_idx : (default_i_start, default_i_end)
	i_start, i_end = min(i_start, i_end), max(i_start, i_end)
	sel = i_start:i_end

	md"""
	**Zoom window:** $(time[i_start]) to $(time[i_end])  
	**Number of days shown:** $(length(sel))
	"""
end

# ╔═╡ 7d6a69f6-6b7d-4c4d-a11b-000000000012
begin
	# ---------------------------
	# Zoomed figure (interactive x-limits)
	# ---------------------------
	pz1 = plot(time[sel], sst[sel], lw = 1.8, label = L"\Theta", color = :black)
	plot!(pz1, time[sel], sst_bg_fixed[sel], lw = 1.5, label = L"\overline{\Theta}")
	plot!(pz1, time[sel], sst_bg_det_core[sel], lw = 1.5, label = L"\overline{\tilde{\Theta}}")
	plot!(pz1, time[sel], sst_bg_glm[sel], lw = 1.5, label = L"\overline{\Theta}_{\mathrm{GLM}}")
	ylabel!(pz1, L"\Theta\ (\mathrm{^\circ C})")
	title!(pz1, "(e) Zoomed background / climatology comparison")
	xlims!(pz1, (time[i_start], time[i_end]))

	pz2 = plot(time[sel], sst_anom_fixed[sel], lw = 1.9, label = L"\Theta'")
	plot!(pz2, time[sel], sst_anom_det[sel], lw = 1.9, label = L"\tilde{\Theta}'")
	plot!(pz2, time[sel], sst_anom_glm[sel], lw = 1.9, label = L"\Theta'_{\mathrm{GLM}}")
	ylabel!(pz2, L"\mathrm{Anomaly\ (^\circ C)}")
	title!(pz2, "(f) Zoomed anomaly comparison")
	xlims!(pz2, (time[i_start], time[i_end]))

	pz3 = plot(time[sel], d_anom_fixed_dt[sel], lw = 1.8, label = L"\partial_t \Theta'")
	plot!(pz3, time[sel], d_anom_det_dt[sel], lw = 1.8, label = L"\partial_t \tilde{\Theta}'")
	plot!(pz3, time[sel], d_anom_glm_dt[sel], lw = 1.8, label = L"\partial_t \Theta'_{\mathrm{GLM}}")
	ylabel!(pz3, L"\mathrm{^\circ C\ day^{-1}}")
	title!(pz3, "(g) Zoomed derivative of anomaly")
	xlims!(pz3, (time[i_start], time[i_end]))

	pz4 = plot(time[sel], dTdt_anom_fixed_true[sel], lw = 1.8, label = L"(\partial_t \Theta)'")
	plot!(pz4, time[sel], dTdt_anom_det_true[sel], lw = 1.8, label = L"(\partial_t \tilde{\Theta})'")
	plot!(pz4, time[sel], dTdt_anom_glm_true[sel], lw = 1.8, label = L"(\partial_t \Theta_{\mathrm{GLM}})'")
	ylabel!(pz4, L"\mathrm{^\circ C\ day^{-1}}")
	title!(pz4, "(h) Zoomed true tendency anomaly")
	xlims!(pz4, (time[i_start], time[i_end]))

	pz5 = plot(time[sel], dTdt_anom_fixed_true[sel] .- d_anom_fixed_dt[sel],
		lw = 1.8, label = L"(\partial_t \Theta)' - \partial_t \Theta'")
	plot!(pz5, time[sel], dTdt_anom_det_true[sel] .- d_anom_det_dt[sel],
		lw = 1.8, label = L"(\partial_t \tilde{\Theta})' - \partial_t \tilde{\Theta}'")
	plot!(pz5, time[sel], dTdt_anom_glm_true[sel] .- d_anom_glm_dt[sel],
		lw = 1.8, label = L"(\partial_t \Theta_{\mathrm{GLM}})' - \partial_t \Theta'_{\mathrm{GLM}}")
	ylabel!(pz5, L"\mathrm{^\circ C\ day^{-1}}")
	title!(pz5, "(i) Zoomed tendency anomaly difference")
	xlims!(pz5, (time[i_start], time[i_end]))

	plot(pz1, pz2, pz3, pz4, pz5, layout = (5,1), size = (1450, 1500))
end

# ╔═╡ 7d6a69f6-6b7d-4c4d-a11b-000000000013
md"""
## Suggested GitHub repository layout

```text
your-repo/
├── Project.toml
├── sst_climatology_explorer.jl
├── ECCOv4r4_Nino3_point_SST_daily_19920101_20171231.nc
└── README.md
```

## Binder launch idea
Use a Binder URL that opens Julia and starts Pluto, or add a small startup script in the repo.
If you want, I can also help you generate:
1. a `README.md`,
2. a Binder startup script, and
3. a `runtime.txt` / `postBuild` setup.
"""

# ╔═╡ Cell order:
# ╠═7d6a69f6-6b7d-4c4d-a11b-000000000001
# ╟─7d6a69f6-6b7d-4c4d-a11b-000000000002
# ╠═7d6a69f6-6b7d-4c4d-a11b-000000000003
# ╠═7d6a69f6-6b7d-4c4d-a11b-000000000004
# ╠═7d6a69f6-6b7d-4c4d-a11b-000000000005
# ╠═7d6a69f6-6b7d-4c4d-a11b-000000000006
# ╠═7d6a69f6-6b7d-4c4d-a11b-000000000007
# ╟─7d6a69f6-6b7d-4c4d-a11b-000000000008
# ╠═7d6a69f6-6b7d-4c4d-a11b-000000000009
# ╠═7d6a69f6-6b7d-4c4d-a11b-000000000010
# ╟─7d6a69f6-6b7d-4c4d-a11b-000000000011
# ╠═7d6a69f6-6b7d-4c4d-a11b-000000000012
# ╟─7d6a69f6-6b7d-4c4d-a11b-000000000013
