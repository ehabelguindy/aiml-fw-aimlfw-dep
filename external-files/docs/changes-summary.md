# Changes Summary

## File: insert_suburban.py  
Original file: external-files/original/insert.py

### Changes made
- Added descriptive header comments explaining:
  - Purpose of the script (SU_MIMO_15m suburban data ingestion)
  - Adaptation from original script
  - Data cleaning behavior (removal of trailing zeros and NaN values)

- Added new dependency:
  - Imported `numpy` for data processing

- Updated dataset input:
  - Changed CSV file from `CellReports.csv` to `SU_MIMO_15m 1.csv`

- Updated InfluxDB configuration:
  - Changed bucket name from `Viavi_Dataset` to `Suburban Dataset`

- Improved documentation within the script:
  - Added inline comments to clarify purpose of dataset and bucket changes

### Reason
- Adapt the original script for suburban traffic data processing
- Improve data quality by handling NaN values and unnecessary zeros
- Ensure correct data source and storage location for suburban dataset
- Enhance readability and maintainability with clearer documentation

---

## File: lstm_traffic_prediction_pipeline_suburban_gpu.py
Original file: external-files/original/pipeline.ipynb

### Changes made
- Converted the original pipeline from a Jupyter notebook (`.ipynb`) into a standalone Python script (`.py`).
- Adapted the pipeline so it can run in the required model/pipeline execution environment instead of depending on notebook execution.
- Changed the pipeline focus to the suburban dataset and updated the training flow to use the `PRB_DL` feature from the feature store.
- Updated the forecasting configuration to use 192 input steps and 192 output steps for 15-minute interval traffic prediction.
- Added GPU-oriented pipeline execution settings and resource configuration for training.
- Added more explicit training, logging, model export, and pipeline upload handling in the Python implementation.

### Reason
- The original notebook format was not suitable for the target model execution workflow.
- A Python script was needed so the pipeline could run properly in the deployment/training environment.
- The pipeline was adapted for suburban traffic prediction requirements and GPU-based execution.

## File: predict-edited.sh
Original file: predict.sh

### Changes made
- Added a shell script header with `#!/bin/bash` and a descriptive top-level comment for the LSTM KServe prediction workflow.
- Renamed and standardized variables:
  - `model_name` → `MODEL_NAME`
  - `file_name` → `INPUT_FILE`
- Updated runtime values:
  - Model name changed from `traffic-forecasting-model` to `lstm-traffic-prediction`
  - Input file changed from `input.json` to `input_suburban.json`
- Added output management:
  - Introduced `OUTPUT_DIR` for prediction results
  - Added timestamp generation with `TIMESTAMP=$(date +"%Y%m%d_%H%M%S")`
  - Ensured the output directory exists with `mkdir -p`
- Replaced the direct cluster IP / ingress-based request flow with local `kubectl port-forward` based access.
- Added a new preprocessing step to start a temporary port-forward to InfluxDB on port `8086`.
- Added execution of `create_input_from_influxdb.py` to automatically fetch the latest 192 data points and generate the model input JSON.
- Added cleanup logic to stop the temporary InfluxDB port-forward process after input generation.
- Added a new KServe predictor port-forward step to expose the predictor service locally on port `8080`.
- Changed the prediction request from a verbose `curl` call with a Host header to a structured `POST` request against `http://localhost:8080/v1/models/${MODEL_NAME}:predict`.
- Added `Content-Type: application/json` and `--max-time 60` to improve API correctness and allow for Knative cold starts.
- Captured the prediction API response into a `RESPONSE` variable for validation and later storage.
- Added response validation by checking whether the returned payload contains `predictions`.
- Added result persistence:
  - Saves responses to `./predictions/`
  - Uses `jq` to wrap the response with timestamp and model metadata when available
  - Falls back to raw JSON saving if `jq` is not installed
- Added user-friendly console logging with numbered steps and success/failure messages.
- Added a quick sanity check that prints the first 5 predicted values from the response.
- Added basic error handling and non-zero exits for failed input creation or failed prediction responses.

### Reason
- Turn the script from a minimal manual inference command into a more complete and reusable prediction workflow.
- Automate input generation from InfluxDB so predictions can be made from recent live data instead of relying on a manually prepared JSON file.
- Improve reliability in KServe/Knative environments by using port-forwarding and allowing time for scale-from-zero startup.
- Make outputs easier to track and review by saving timestamped prediction results.
- Improve maintainability and usability with clearer logging, structured variables, and better error handling.
