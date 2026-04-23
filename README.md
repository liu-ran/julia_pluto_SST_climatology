# Julia Binder + Pluto SST explorer

This repo contains a Pluto notebook version of the SST climatology/anomaly analysis.

## Files
- `sst_climatology_explorer.jl`: Pluto notebook
- `Project.toml`: Julia dependencies for Binder/local use
- `ECCOv4r4_Nino3_point_SST_daily_19920101_20171231.nc`: input NetCDF data file

## Main interactive feature
The zoomed figure is controlled by a slider that changes the x-range of the zoom section.

## Local run
In Julia:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
using Pluto
Pluto.run()
```

Then open `sst_climatology_explorer.jl`.

## Binder suggestion
A practical approach is to use Binder with Julia + Pluto. Typical repo contents for that include:

- `Project.toml`
- notebook file
- optional `.binder/` startup files

If needed, add:

`.binder/start_pluto.jl`
```julia
using Pluto
Pluto.run(host="0.0.0.0", port=1234, launch_browser=false)
```

`.binder/postBuild`
```bash
#!/bin/bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

Then configure Binder to start Pluto.
