from ecmwf.opendata import Client

import xarray as xr
import numpy as np
import json
import os
import time
import requests
from datetime import datetime, timedelta

print("=" * 60)
print("MJO RMM Forecast Pipeline")
print("=" * 60)
print()

# --- Paths ---
OBS_FILE   = "ncep_eofs.nc"    # NCEP/WH2004 EOF patterns — committed to repo
CLIMO_FILE = "ERA5_climo.nc"   # ERA5 climatology — committed to repo
OUTPUT     = "forecast_rmm.json"

# Forecast steps 0 to 240h every 24h
STEPS = list(range(0, 241, 24))

# --- Step 1: Verify local NetCDF files are present ---
# Both files are committed to the repo — no download needed.
print("Step 1: Checking local NetCDF files...")
for fname in [OBS_FILE, CLIMO_FILE]:
    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"{fname} not found. Make sure it is committed to the repo root."
        )
    print(f"  {fname} found ({os.path.getsize(fname)/1024:.1f} KB)")

# --- Step 2: Load EOF patterns from NCEP file ---
print()
print("Step 2: Loading EOF patterns from NCEP file...")
obs_ds = xr.open_dataset(OBS_FILE)

eof1_olr  = obs_ds["eof1_olr"].values
eof2_olr  = obs_ds["eof2_olr"].values
eof1_u850 = obs_ds["eof1_u850"].values
eof2_u850 = obs_ds["eof2_u850"].values
eof1_u200 = obs_ds["eof1_u200"].values
eof2_u200 = obs_ds["eof2_u200"].values
eof_lons  = obs_ds["lon1deg"].values   # 0–359 at 1-degree resolution

# Read normalisation scalars from file attributes (NCEP/WH2004 values)
OLR_SCALAR  = float(obs_ds.attrs["olr_scalar"])
U850_SCALAR = float(obs_ds.attrs["u850_scalar"])
U200_SCALAR = float(obs_ds.attrs["u200_scalar"])
RMM1_STD    = float(obs_ds.attrs["rmm1_std"])
RMM2_STD    = float(obs_ds.attrs["rmm2_std"])

print(f"  EOF patterns loaded. Longitude points: {len(eof_lons)}")
print(f"  Scalars — OLR: {OLR_SCALAR}  U850: {U850_SCALAR}  U200: {U200_SCALAR}")
print(f"  RMM std — RMM1: {RMM1_STD:.4f}  RMM2: {RMM2_STD:.4f}")
obs_ds.close()

# --- Step 3: Load ERA5 climatology ---
print("Step 3: Loading ERA5 climatology...")
climo_check = xr.open_dataset(CLIMO_FILE)
print(f"  Variables: {list(climo_check.data_vars)}")
climo_check.close()
print()

# --- Step 4: Download ECMWF forecast step by step ---
print("Step 4: Downloading ECMWF forecast data...")
client = Client(source="ecmwf")

def download_with_retry(params, target, max_retries=3):
    for attempt in range(max_retries):
        try:
            if os.path.exists(target):
                os.remove(target)
            client.retrieve(**params, target=target)
            return True
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {str(e)[:80]}")
            if attempt < max_retries - 1:
                print(f"  Retrying in 5 seconds...")
                time.sleep(5)
    return False

print("  U wind (850+200 hPa)...")
for step in STEPS:
    fname = f"fcst_u_{step:03d}.grib2"
    if os.path.exists(fname):
        print(f"  +{step:3d}h cached")
        continue
    print(f"  +{step:3d}h...", end=" ", flush=True)
    ok = download_with_retry({
        "type": "fc", "param": "u",
        "levtype": "pl", "levelist": [850, 200], "step": step,
    }, fname)
    print("OK" if ok else "FAILED")

print("  TTR (OLR)...")
for step in STEPS:
    fname = f"fcst_ttr_{step:03d}.grib2"
    if os.path.exists(fname):
        print(f"  +{step:3d}h cached")
        continue
    print(f"  +{step:3d}h...", end=" ", flush=True)
    ok = download_with_retry({
        "type": "fc", "param": "ttr",
        "levtype": "sfc", "step": step,
    }, fname)
    print("OK" if ok else "FAILED")

print()

# --- Step 5: Get model init time and ECMWF lon grid ---
print("Step 5: Computing RMM indices...")
print()

ds_u0 = xr.open_dataset(
    "fcst_u_000.grib2", engine="cfgrib",
    backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}}
)
init_time = ds_u0.time.values
init_dt = datetime.utcfromtimestamp(
    (init_time - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
)
ecmwf_lons = ds_u0["longitude"].values
ecmwf_lons_360 = ecmwf_lons % 360
sort_idx = np.argsort(ecmwf_lons_360)
ds_u0.close()

print(f"  Model init: {init_dt.strftime('%Y-%m-%d %H:%M UTC')}")
print()

climo_ds = xr.open_dataset(CLIMO_FILE)
rmm_results = []

for i in range(1, len(STEPS)):
    step      = STEPS[i]
    step_prev = STEPS[i - 1]

    u_file        = f"fcst_u_{step:03d}.grib2"
    ttr_file      = f"fcst_ttr_{step:03d}.grib2"
    ttr_prev_file = f"fcst_ttr_{step_prev:03d}.grib2"

    if not all(os.path.exists(f) for f in [u_file, ttr_file, ttr_prev_file]):
        print(f"  Skipping +{step}h — missing files")
        continue

    ds_u = xr.open_dataset(
        u_file, engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}}
    )
    ds_ttr = xr.open_dataset(
        ttr_file, engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": "ttr"}}
    )
    ds_ttr_prev = xr.open_dataset(
        ttr_prev_file, engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": "ttr"}}
    )

    valid_dt = init_dt + timedelta(hours=step)
    doy = valid_dt.timetuple().tm_yday

    # Zonal means over 15S–15N
    u850_zm = ds_u["u"].sel(
        isobaricInhPa=850, latitude=slice(15, -15)
    ).mean(dim="latitude").values

    u200_zm = ds_u["u"].sel(
        isobaricInhPa=200, latitude=slice(15, -15)
    ).mean(dim="latitude").values

    olr_zm = -((ds_ttr["ttr"].sel(latitude=slice(15, -15)) -
                ds_ttr_prev["ttr"].sel(latitude=slice(15, -15))
               ) / 86400.0).mean(dim="latitude").values

    # Interpolate ECMWF grid → EOF lon grid (0–359, 1-degree)
    u850_interp = np.interp(eof_lons, ecmwf_lons_360[sort_idx], u850_zm[sort_idx])
    u200_interp = np.interp(eof_lons, ecmwf_lons_360[sort_idx], u200_zm[sort_idx])
    olr_interp  = np.interp(eof_lons, ecmwf_lons_360[sort_idx], olr_zm[sort_idx])

    # Remove climatology
    climo_day  = climo_ds.sel(dayofyear=doy)
    u850_anom  = u850_interp - climo_day["uwnd850"].values
    u200_anom  = u200_interp - climo_day["uwnd200"].values
    olr_anom   = olr_interp  - climo_day["olr"].values

    # Normalise by NCEP scalars
    u850_anom_norm = u850_anom / U850_SCALAR
    u200_anom_norm = u200_anom / U200_SCALAR
    olr_anom_norm  = olr_anom  / OLR_SCALAR

    # Project onto EOFs — divide by separate RMM1/RMM2 std (WH2004)
    rmm1 = (np.dot(olr_anom_norm,  eof1_olr)  +
            np.dot(u850_anom_norm, eof1_u850) +
            np.dot(u200_anom_norm, eof1_u200)) / RMM1_STD

    rmm2 = (np.dot(olr_anom_norm,  eof2_olr)  +
            np.dot(u850_anom_norm, eof2_u850) +
            np.dot(u200_anom_norm, eof2_u200)) / RMM2_STD

    amplitude = float(np.sqrt(rmm1**2 + rmm2**2))
    angle     = np.degrees(np.arctan2(rmm2, rmm1))
    phase     = int(((angle + 180) // 45) % 8) + 1
    phase_label = "weak" if amplitude < 1.0 else str(phase)

    result = {
        "date":        valid_dt.strftime("%Y-%m-%d"),
        "step_hours":  step,
        "rmm1":        round(float(rmm1), 4),
        "rmm2":        round(float(rmm2), 4),
        "amplitude":   round(amplitude, 4),
        "phase":       phase,
        "phase_label": phase_label,
    }
    rmm_results.append(result)

    print(f"  +{step:3d}h {valid_dt.strftime('%Y-%m-%d')} "
          f"RMM1={rmm1:+.3f} RMM2={rmm2:+.3f} "
          f"Amp={amplitude:.3f} Phase={phase_label}")

    ds_u.close()
    ds_ttr.close()
    ds_ttr_prev.close()

climo_ds.close()
print()

# --- Step 6: Save forecast JSON ---
print("Step 6: Saving forecast_rmm.json...")
output = {
    "model":           "ECMWF IFS",
    "eof_source":      "NCEP/WH2004 (ncep_eofs.nc)",
    "init_time":       init_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "generated_utc":   datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "n_forecast_days": len(rmm_results),
    "forecast":        rmm_results,
}

with open(OUTPUT, 'w') as f:
    json.dump(output, f, indent=2)

print(f"  Saved {OUTPUT} — {len(rmm_results)} forecast days")
print()
print("=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
