#!/usr/bin/env python3
"""
Fetches the last 192 PRB_DL values from InfluxDB, normalizes them using
the training scaler, and writes input_suburban.json for KServe prediction.
"""

import json
import os
import numpy as np
from influxdb_client import InfluxDBClient

# InfluxDB connection — override via env vars if needed
INFLUXDB_URL   = f"http://{os.getenv('INFLUXDB_IP', 'localhost')}:{os.getenv('INFLUXDB_PORT', '8086')}"
INFLUXDB_TOKEN = os.getenv('INFLUXDB_TOKEN', 'Dl8FVVIaYMUhJoG1kSmZ')
ORG            = os.getenv('INFLUXDB_ORG', 'primary')
BUCKET         = os.getenv('INFLUXDB_BUCKET', 'Suburban_Dataset')
MEASUREMENT    = os.getenv('INFLUXDB_MEASUREMENT', 'cell_metrics')
CELL_NAME      = os.getenv('CELL_NAME', 'BRANGES_T1')

OUTPUT_FILE = 'input_suburban.json'
NUM_STEPS   = 192  # 2 days @ 15-min intervals

# Step 1: Query the last 192 PRB_DL values from InfluxDB in chronological order
print(f"Querying InfluxDB ({INFLUXDB_URL}) — last {NUM_STEPS} PRB_DL values for {CELL_NAME}...")
try:
    client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=ORG, timeout=30000)
    result = client.query_api().query(f'''
        from(bucket: "{BUCKET}")
          |> range(start: -365d)
          |> filter(fn: (r) => r["_measurement"] == "{MEASUREMENT}")
          |> filter(fn: (r) => r["CELLULE"] == "{CELL_NAME}")
          |> filter(fn: (r) => r["_field"] == "PRB_DL")
          |> sort(columns: ["_time"], desc: true)
          |> limit(n: {NUM_STEPS})
          |> sort(columns: ["_time"], desc: false)
    ''')
    client.close()

    values = [
        float(r.get_value())
        for table in result
        for r in table.records
        if r.get_value() is not None and not np.isnan(r.get_value())
    ]
except Exception as e:
    print(f"❌ InfluxDB query failed: {e}")
    exit(1)

if len(values) < NUM_STEPS:
    print(f"❌ Not enough data: got {len(values)}, need {NUM_STEPS}")
    exit(1)

print(f"✓ Got {len(values)} values  (range: {min(values):.2f} – {max(values):.2f})")

# Step 2: Normalize to [0, 1] — same formula as training pipeline
arr       = np.array(values)
min_val   = float(arr.min())
range_val = float(arr.max() - arr.min()) or 1.0
normalized = ((arr - min_val) / range_val).tolist()
print(f"✓ Normalized  (min={min_val:.2f}, range={range_val:.2f})")

# Step 3: Write JSON in TF Serving format — shape: (1, 192, 1)
json.dump(
    {"signature_name": "serving_default", "instances": [[[v] for v in normalized]]},
    open(OUTPUT_FILE, 'w'),
    indent=2
)

print(f"✓ Written {OUTPUT_FILE}  ({NUM_STEPS} steps, normalized range: {min(normalized):.4f} – {max(normalized):.4f})")
