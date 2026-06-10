import requests
import json
from datetime import datetime

BOM_URL = 'https://www.bom.gov.au/clim_data/IDCKGEM000/rmm.74toRealtime.txt'
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
OUTPUT  = 'observed_rmm.json'
DAYS    = 40

def parse_bom(text):
    rows = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('RMM') or line.startswith('year'):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            year  = int(parts[0])
            month = int(parts[1])
            day   = int(parts[2])
            rmm1  = float(parts[3])
            rmm2  = float(parts[4])
            phase = int(parts[5])
            amp   = float(parts[6])
        except (ValueError, IndexError):
            continue
        # Skip missing values
        if abs(rmm1) > 900 or abs(rmm2) > 900:
            continue
        rows.append({
            'date':      f'{day:02d}-{datetime(year,month,day).strftime("%b")}-{str(year)[-2:]}',
            'year':      year,
            'month':     month,
            'day':       day,
            'rmm1':      round(rmm1, 4),
            'rmm2':      round(rmm2, 4),
            'amplitude': round(amp,  4),
            'phase':     phase,
        })
    return rows

def main():
    print('Fetching BOM RMM data...')
    r = requests.get(BOM_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    print(f'  Downloaded {len(r.text):,} characters')

    rows = parse_bom(r.text)
    print(f'  Parsed {len(rows)} valid data points')

    last40 = rows[-DAYS:]
    print(f'  Last {DAYS} days: {last40[0]["date"]} to {last40[-1]["date"]}')

    output = {
        'source':     'BOM RMM — https://www.bom.gov.au/clim_data/IDCKGEM000/rmm.74toRealtime.txt',
        'fetched_utc': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'n_days':      len(last40),
        'observed':    last40,
    }

    with open(OUTPUT, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'  Saved {OUTPUT}')

if __name__ == '__main__':
    main()
