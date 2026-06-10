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
OBS_FILE   = "MJO_obs_created.nc"
CLIMO_FILE = "ERA5_climo.nc"
OUTPUT     = "forecast_rmm.json"

# Normalization scalars (from ERA5 filtered anomaly file)
OLR_SCALAR  = 14.533773
U850_SCALAR = 1.829570
U200_SCALAR = 5.159616
EOF_SCALE   = 11.709

# Forecast steps 0 to 240h every 24h
STEPS = list(range(0, 241, 24))

# --- Step 1: Download NetCDF files if not present ---
NC_URLS = {
    OBS_FILE:   'https://github.com/WillyChap/MJOcast/raw/main/MJOcast/Observations/MJO_obs_created.nc',
    CLIMO_FILE: 'https://github.com/WillyChap/MJOcast/raw/main/MJOcast/Observations/ERA5_climo.nc',
}

for fname, url in NC_URLS.items():
    if not os.path.exists(fname):
        print(f"Downloading {fname}...")
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(fname, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                f.write(chunk)
        print(f"  Done — {os.path.getsize(fname)/1024/1024:.1f} MB")
    else:
        print(f"  {fname} already present")

# --- Step 2: Load EOF patterns ---
print()
print("Step 2: Loading EOF patterns...")
obs_ds    = xr.open_dataset(OBS_FILE)
eof1_olr  = obs_ds["eof1_olr"].values
eof2_olr  = obs_ds["eof2_olr"].values
eof1_u850 = obs_ds["eof1_u850"].values
eof2_u850 = obs_ds["eof2_u850"].values
eof1_u200 = obs_ds["eof1_u200"].values
eof2_u200 = obs_ds["eof2_u200"].values
eof_lons  = obs_ds["longitude"].values
print(f"  EOF patterns loaded. Longitude points: {len(eof_lons)}")
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
            print(f"    Attempt {attempt+1} failed: {str(e)[:80]}")
            if attempt < max_retries - 1:
                print(f"    Retrying in 5 seconds...")
                time.sleep(5)
    return False

print("  U wind (850+200 hPa)...")
for step in STEPS:
    fname = f"fcst_u_{step:03d}.grib2"
    if os.path.exists(fname):
        print(f"    +{step:3d}h cached")
        continue
    print(f"    +{step:3d}h...", end=" ", flush=True)
    ok = download_with_retry({
        "type": "fc", "param": "u",
        "levtype": "pl", "levelist": [850, 200], "step": step,
    }, fname)
    print("OK" if ok else "FAILED")

print("  TTR (OLR)...")
for step in STEPS:
    fname = f"fcst_ttr_{step:03d}.grib2"
    if os.path.exists(fname):
        print(f"    +{step:3d}h cached")
        continue
    print(f"    +{step:3d}h...", end=" ", flush=True)
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
ecmwf_lons     = ds_u0["longitude"].values
ecmwf_lons_360 = ecmwf_lons % 360
sort_idx       = np.argsort(ecmwf_lons_360)
ds_u0.close()

print(f"  Model init: {init_dt.strftime('%Y-%m-%d %H:%M UTC')}")
print()

climo_ds    = xr.open_dataset(CLIMO_FILE)
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
    doy      = valid_dt.timetuple().tm_yday

    u850_zm = ds_u["u"].sel(
        isobaricInhPa=850, latitude=slice(15, -15)
    ).mean(dim="latitude").values

    u200_zm = ds_u["u"].sel(
        isobaricInhPa=200, latitude=slice(15, -15)
    ).mean(dim="latitude").values

    olr_zm = -((ds_ttr["ttr"].sel(latitude=slice(15, -15)) -
                ds_ttr_prev["ttr"].sel(latitude=slice(15, -15))
               ) / 86400.0).mean(dim="latitude").values

    u850_interp = np.interp(eof_lons, ecmwf_lons_360[sort_idx], u850_zm[sort_idx])
    u200_interp = np.interp(eof_lons, ecmwf_lons_360[sort_idx], u200_zm[sort_idx])
    olr_interp  = np.interp(eof_lons, ecmwf_lons_360[sort_idx], olr_zm[sort_idx])

    climo_day = climo_ds.sel(dayofyear=doy)
    u850_anom = u850_interp - climo_day["uwnd850"].values
    u200_anom = u200_interp - climo_day["uwnd200"].values
    olr_anom  = olr_interp  - climo_day["olr"].values

    u850_anom_norm = u850_anom / U850_SCALAR
    u200_anom_norm = u200_anom / U200_SCALAR
    olr_anom_norm  = olr_anom  / OLR_SCALAR

    rmm1 = (np.dot(olr_anom_norm,  eof1_olr)  +
            np.dot(u850_anom_norm, eof1_u850) +
            np.dot(u200_anom_norm, eof1_u200)) / EOF_SCALE

    rmm2 = (np.dot(olr_anom_norm,  eof2_olr)  +
            np.dot(u850_anom_norm, eof2_u850) +
            np.dot(u200_anom_norm, eof2_u200)) / EOF_SCALE

    amplitude   = float(np.sqrt(rmm1**2 + rmm2**2))
    angle       = np.degrees(np.arctan2(rmm2, rmm1))
    phase       = int(((angle + 180) // 45) % 8) + 1
    phase_label = "weak" if amplitude < 1.0 else str(phase)

    result = {
        "date":        valid_dt.strftime("%Y-%m-%d"),
        "step_hours":  step,
        "rmm1":        round(float(rmm1), 4),
        "rmm2":        round(float(rmm2), 4),
        "amplitude":   round(amplitude,   4),
        "phase":       phase,
        "phase_label": phase_label,
    }
    rmm_results.append(result)

    print(f"  +{step:3d}h  {valid_dt.strftime('%Y-%m-%d')}  "
          f"RMM1={rmm1:+.3f}  RMM2={rmm2:+.3f}  "
          f"Amp={amplitude:.3f}  Phase={phase_label}")

    ds_u.close()
    ds_ttr.close()
    ds_ttr_prev.close()

climo_ds.close()
print()

# --- Step 6: Save forecast JSON ---
print("Step 6: Saving forecast_rmm.json...")
output = {
    "model":           "ECMWF IFS",
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
