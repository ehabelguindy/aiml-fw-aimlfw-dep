#!/bin/bash
# LSTM Traffic Prediction via KServe

MODEL_NAME="lstm-traffic-prediction"
INPUT_FILE="input_suburban.json"
OUTPUT_DIR="./predictions"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "$OUTPUT_DIR"

# Starts a temporary port-forward to InfluxDB so create_input_from_influxdb.py
# can reach it at localhost:8086 without any manual setup beforehand.
echo "[$(date '+%H:%M:%S')] Step 1: Starting port-forward to InfluxDB..."
lsof -ti:8086 | xargs kill -9 2>/dev/null || true
kubectl port-forward -n default svc/my-release-influxdb 8086:8086 > /dev/null 2>&1 &
INFLUX_PF_PID=$!
sleep 5

# Reads the last 192 data points (2 days of 15-min intervals) from InfluxDB and
# writes them as a normalized JSON array into input_suburban.json, ready for the model.
echo "[$(date '+%H:%M:%S')] Step 2: Fetching data from InfluxDB..."
if ! python3 create_input_from_influxdb.py 2>&1; then
    kill $INFLUX_PF_PID 2>/dev/null
    echo "❌ Failed to create input from InfluxDB."
    exit 1
fi

kill $INFLUX_PF_PID 2>/dev/null

# Opens a temporary tunnel from localhost:8080 to the KServe predictor service (port 80).
# This works even when the model is scaled to zero — Knative will spin up a pod on demand.
echo "[$(date '+%H:%M:%S')] Step 3: Starting port-forward to KServe predictor..."
lsof -ti:8080 | xargs kill -9 2>/dev/null || true
kubectl port-forward -n kserve-test svc/${MODEL_NAME}-predictor 8080:80 > /dev/null 2>&1 &
PF_PID=$!
sleep 5

# Posts the input JSON to TensorFlow Serving's v1 predict API.
# The model returns 192 predicted PRB_DL values (next 2 days at 15-min intervals).
# --max-time 60 gives Knative time to cold-start the pod if scaled to zero.
echo "[$(date '+%H:%M:%S')] Step 4: Sending prediction request..."
RESPONSE=$(curl -s --max-time 60 -X POST "http://localhost:8080/v1/models/${MODEL_NAME}:predict" \
    -H "Content-Type: application/json" \
    -d @"$INPUT_FILE")

kill $PF_PID 2>/dev/null

# Saves the full response with timestamp and model name to ./predictions/.
# Prints the first 5 predicted values as a quick sanity check.
if echo "$RESPONSE" | grep -q "predictions"; then
    OUTPUT_FILE="$OUTPUT_DIR/prediction_${TIMESTAMP}.json"
    if command -v jq &>/dev/null; then
        echo "$RESPONSE" | jq --arg ts "$(date -Iseconds)" --arg m "$MODEL_NAME" \
            '{timestamp: $ts, model: $m, response: .}' > "$OUTPUT_FILE"
    else
        echo "$RESPONSE" > "$OUTPUT_FILE"
    fi

    echo "✅ Prediction saved: $OUTPUT_FILE"
    echo "First 5 values:"
    echo "$RESPONSE" | jq -r '.predictions[0][:5][]' 2>/dev/null
else
    echo "❌ Prediction failed:"
    echo "$RESPONSE"
    exit 1
fi
