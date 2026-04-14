import csv
import logging
from datetime import datetime
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# InfluxDB connection settings
DATASET_PATH = 'CellReports.csv'
INFLUXDB_IP = 'localhost'  # Using port-forward to access InfluxDB
INFLUXDB_TOKEN = 'Dl8FVVIaYMUhJoG1kSmZ'  # Token from my-release-influxdb secret

org = "primary"
bucket = "Viavi_Dataset"
url = f"http://{INFLUXDB_IP}:8086"

try:
    # Connect to InfluxDB
    client = InfluxDBClient(url=url, token=INFLUXDB_TOKEN, org=org)
    write_api = client.write_api(write_options=SYNCHRONOUS)

    # Read CSV file and write to InfluxDB
    with open(DATASET_PATH, 'r') as file:
        csv_reader = csv.DictReader(file)
        total_rows = sum(1 for row in csv.DictReader(open(DATASET_PATH)))

        for index, row in enumerate(csv_reader, 1):
            try:
                # Convert timestamp to nanoseconds
                timestamp = int(datetime.fromtimestamp(int(row['timestamp'])).timestamp() * 1e9)

                # Create Point
                point = Point("cell_metrics")

                # Add tag
                point.tag("Viavi.Cell.Name", row['Viavi.Cell.Name'])

                # Add fields (excluding timestamp and Viavi.Cell.Name)
                for key, value in row.items():
                    if key not in ['timestamp', 'Viavi.Cell.Name']:
                        # Skip None, empty, or whitespace-only values
                        if value is None or (isinstance(value, str) and value.strip() == ''):
                            continue
                        try:
                            float_value = float(value)
                            point.field(key, float_value)
                        except (ValueError, TypeError):
                            logging.warning(f"Row {index}: Unable to convert field '{key}' to float. Value: {value}")

                # Set timestamp
                point.time(timestamp)

                # Write point to InfluxDB
                write_api.write(bucket=bucket, org=org, record=point)

                # Display progress (every 100 rows)
                if index % 100 == 0:
                    logging.info(f"Progress: {index}/{total_rows} rows processed ({index/total_rows*100:.2f}%)")

            except Exception as e:
                logging.error(f"Error processing row {index}: {e}")

    logging.info("Data import completed.")

except Exception as e:
    logging.error(f"Error during script execution: {e}")

finally:
    # Close client
    client.close()
    logging.info("InfluxDB connection closed.")