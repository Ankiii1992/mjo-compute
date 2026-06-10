import json
import os
from datetime import datetime

OBSERVED_FILE = "observed_rmm.json"
FORECAST_FILE = "forecast_rmm.json"
OUTPUT_FILE   = "data/mjo_data.json"

def main():
    print("Combining observed + forecast into mjo_data.json...")

    # Load observed
    with open(OBSERVED_FILE) as f:
        obs = json.load(f)
    print(f"  Observed: {obs['n_days']} days, last = {obs['observed'][-1]['date']}")

    # Load forecast
    with open(FORECAST_FILE) as f:
        fcst = json.load(f)
    print(f"  Forecast: {fcst['n_forecast_days']} days, init = {fcst['init_time']}")

    # Build combined output
    output = {
        "meta": {
            "generated_utc":    datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "observed_source":  "BOM RMM — bom.gov.au",
            "forecast_source":  "ECMWF IFS open data",
            "eof_source":       "MJOcast ERA5 Wheeler-Hendon EOFs",
            "model_init":       fcst["init_time"],
            "n_observed_days":  obs["n_days"],
            "n_forecast_days":  fcst["n_forecast_days"],
        },
        "observed": obs["observed"],
        "forecast": fcst["forecast"],
    }

    # Ensure data/ folder exists
    os.makedirs("data", exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"  Saved {OUTPUT_FILE}")
    print(f"  Observed days: {len(output['observed'])}")
    print(f"  Forecast days: {len(output['forecast'])}")
    print("Done!")

if __name__ == '__main__':
    main()
