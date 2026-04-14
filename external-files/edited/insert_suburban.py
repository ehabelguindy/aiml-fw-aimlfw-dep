#!/usr/bin/env python3
# ==================================================================================
#
#       Insert SU_MIMO_15m Suburban Data into InfluxDB
#       Adapted from insert.py for suburban traffic data.
#
#       PRB_DL may legitimately be 0 (e.g. no traffic); we do NOT drop zeros.
#       We only skip NaN and trim trailing NaN padding at the end of the series.
#
# ==================================================================================

import csv
import logging
from datetime import datetime
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import numpy as np

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------------------------
# Configuration: CSV source, InfluxDB URL/token/org/bucket
# Use port-forward: kubectl port-forward -n default svc/my-release-influxdb 8086:8086
# ---------------------------------------------------------------------------
DATASET_PATH = 'SU_MIMO_15m 1.csv'  # Semicolon-separated suburban export
INFLUXDB_IP = 'localhost'
INFLUXDB_TOKEN = 'Dl8FVVIaYMUhJoG1kSmZ'  # From my-release-influxdb secret / UI

org = "primary"
bucket = "Suburban_Dataset"
url = f"http://{INFLUXDB_IP}:8086"

try:
    # -----------------------------------------------------------------------
    # Connect: synchronous writes so each point is flushed before the next.
    # -----------------------------------------------------------------------
    client = InfluxDBClient(url=url, token=INFLUXDB_TOKEN, org=org, timeout=30000)
    write_api = client.write_api(write_options=SYNCHRONOUS)

    # =======================================================================
    # Step 1: Read CSV and build row list for PRB_DL time series
    # =======================================================================
    # We read PRB_DL as the anchor column. NaN is skipped or ends the series
    # if it appears after real data (trailing padding). Zero is kept — it is
    # valid traffic/load data.
    # =======================================================================
    print("=" * 60)
    print("Step 1: Reading and cleaning data")
    print("=" * 60)

    all_rows = []
    prb_dl_values = []

    with open(DATASET_PATH, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file, delimiter=';')
        for row in csv_reader:
            prb_dl_str = row.get('PRB_DL', '').strip()
            if not prb_dl_str:
                continue

            try:
                prb_dl_val = float(prb_dl_str)
            except (ValueError, TypeError):
                continue

            # Drop only invalid floats: NaN. Do NOT treat 0 as "no data".
            if np.isnan(prb_dl_val):
                if len(all_rows) > 0:
                    # Trailing NaN after real points — stop (padding at end of file)
                    break
                continue

            all_rows.append(row)
            prb_dl_values.append(prb_dl_val)

    # Trim only trailing NaNs from the end (if any slipped through)
    last_valid_idx = len(prb_dl_values)
    for i in range(len(prb_dl_values) - 1, -1, -1):
        if not np.isnan(prb_dl_values[i]):
            last_valid_idx = i + 1
            break

    all_rows = all_rows[:last_valid_idx]
    prb_dl_values = prb_dl_values[:last_valid_idx]

    print(f"✓ Read {len(all_rows)} rows with valid PRB_DL data (zeros kept)")
    print(f"✓ Trimmed trailing NaN only (not zeros)")
    if prb_dl_values:
        print(f"✓ PRB_DL range: [{min(prb_dl_values):.2f}, {max(prb_dl_values):.2f}]")

    # =======================================================================
    # Step 2: Write each row as one InfluxDB point (line protocol → bucket)
    # =======================================================================
    # One Point per row: measurement cell_metrics, tag CELLULE, fields from
    # numeric columns, time from MINIMALE(PSDATE).
    # =======================================================================
    print("\n" + "=" * 60)
    print("Step 2: Writing data to InfluxDB")
    print("=" * 60)
    print(f"Target bucket: {bucket}")
    print("=" * 60)

    total_rows = len(all_rows)
    successful_writes = 0
    failed_writes = 0

    for index, row in enumerate(all_rows, 1):
        try:
            # Timestamp: MINIMALE(PSDATE) as "DD/MM/YYYY HH:MM:SS" → nanoseconds
            date_str = row['MINIMALE(PSDATE)']
            dt = datetime.strptime(date_str, "%d/%m/%Y %H:%M:%S")
            timestamp = int(dt.timestamp() * 1e9)

            point = Point("cell_metrics")
            point.tag("CELLULE", row['CELLULE'])

            # All numeric fields except identifiers and the time column
            excluded_keys = {'CODE_ELT_CELLULE', 'CELLULE', 'MINIMALE(PSDATE)', ''}

            for key, value in row.items():
                if key in excluded_keys:
                    continue
                if value is None or (isinstance(value, str) and value.strip() == ''):
                    continue

                try:
                    float_value = float(value)
                    if np.isnan(float_value):
                        continue
                    # Zero is written as a valid field value
                    point.field(key, float_value)
                except (ValueError, TypeError):
                    pass

            point.time(timestamp)
            write_api.write(bucket=bucket, org=org, record=point)
            successful_writes += 1

            if index % 100 == 0:
                logging.info(
                    f"Progress: {index}/{total_rows} rows ({index/total_rows*100:.2f}%)"
                )

        except Exception as e:
            failed_writes += 1
            logging.error(f"Error processing row {index}: {e}")
            if index <= 5:
                logging.error(f"Row data: {row}")

    print("\n" + "=" * 60)
    print("✅ Data import completed!")
    print("=" * 60)
    print(f"Total rows processed: {total_rows}")
    print(f"Successful writes: {successful_writes}")
    print(f"Failed writes: {failed_writes}")

    if failed_writes > 0:
        print(f"\n⚠️  {failed_writes} rows failed to write")
        print("   Common causes:")
        print("   - Bucket 'Suburban_Dataset' doesn't exist")
        print("   - InfluxDB connection issues")
    else:
        print("\n✅ All rows written successfully!")

except Exception as e:
    logging.error(f"❌ Error during script execution: {e}")
    import traceback
    traceback.print_exc()

finally:
    if 'client' in locals():
        client.close()
        logging.info("InfluxDB connection closed.")
