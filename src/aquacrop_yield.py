"""Grain yield of an irrigation schedule with AquaCrop-OSPy, using the maize parameters calibrated on the Bushland
lysimeters (notebook 01 v2): WP 33.7, Kcb 1.05, Zmax 1.2 m, MaturityCD 125, subsurface-drip wetted surface 1%.
Soil: one layer from SoilGrids water limits (theta_fc, theta_wp; saturation theta_fc + 0.10, Ksat 40 mm/d assumed).
Weather: NASA POWER daily Tmin, Tmax, rain and FAO-56 ET0. Colab only (needs the aquacrop package)."""
import pandas as pd

CAL = dict(WP=33.7, Kcb=1.05, Zmax=1.2, MaturityCD=125)
DZ = [0.1] * 15 + [0.15] * 6


def season_yield(daily, soil, sow_date, irr_mm, wet_surf=1, params=CAL):
    """daily: DataFrame (date index) with tmin, tmax, rain_mm, et0_mm. irr_mm: Series of daily net irrigation (mm,
    already after drip efficiency) indexed by date. Returns dry grain yield (t/ha), seasonal irrigation (mm) and ETa (mm)."""
    from aquacrop import AquaCropModel, Crop, InitialWaterContent, IrrigationManagement, Soil
    sow = pd.Timestamp(sow_date); end = min(sow + pd.Timedelta(days=200), daily.index[-1])   # weather ends 2024-12-31
    w = daily.loc[sow:end]
    wdf = pd.DataFrame({"MinTemp": w.tmin.values, "MaxTemp": w.tmax.values, "Precipitation": w.rain_mm.values,
                        "ReferenceET": w.et0_mm.values, "Date": w.index})
    s = Soil("custom", dz=DZ)
    s.add_layer(thickness=s.zSoil, thWP=soil["theta_wp"], thFC=soil["theta_fc"], thS=min(soil["theta_fc"] + 0.10, 0.55),
                Ksat=40, penetrability=100)
    irr = irr_mm[irr_mm > 0]
    sched = pd.DataFrame({"Date": pd.to_datetime(irr.index), "Depth": irr.values})
    model = AquaCropModel(sim_start_time=sow.strftime("%Y/%m/%d"), sim_end_time=end.strftime("%Y/%m/%d"), weather_df=wdf, soil=s,
                          crop=Crop("Maize", planting_date=sow.strftime("%m/%d"), **params),
                          initial_water_content=InitialWaterContent(value=["FC"]),          # field capacity at sowing, as in the twin
                          irrigation_management=IrrigationManagement(irrigation_method=3, Schedule=sched, WetSurf=wet_surf))
    model.run_model(till_termination=True)
    res = model.get_simulation_results()
    ycol = [c for c in res.columns if "dry yield" in c.lower()] or [c for c in res.columns if "yield" in c.lower() and "potential" not in c.lower()]
    flux = model._outputs.water_flux
    flux = flux[flux["dap"] > 0]
    return dict(yield_t_ha=float(res[ycol[0]].iloc[0]), irrigation_mm=float(irr.sum()), eta_mm=float((flux.Es + flux.Tr).sum()))
