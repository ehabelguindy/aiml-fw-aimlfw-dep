#!/bin/bash
# ==================================================================================
# Register Suburban Model at MME (Model Management Service)
# ==================================================================================

# Override with: MME_IP=10.237.129.136 MME_PORT=32006 ./register_model_suburban.sh
MME_IP="${MME_IP:-localhost}"
MME_PORT="${MME_PORT:-32006}"
API_URL="http://${MME_IP}:${MME_PORT}/ai-ml-model-registration/v1/model-registrations"

# S3/MinIO-safe name (no underscores): MMS uses modelName as the S3 bucket on upload.
MODEL_NAME="suburban-lstm-prbdl"
MODEL_VERSION="1"

echo "=========================================="
echo "Registering Suburban Model at MME"
echo "=========================================="
echo ""
echo "Model: ${MODEL_NAME}/${MODEL_VERSION}"
echo "Endpoint: ${API_URL}"
echo ""

RESPONSE=$(curl -s -X POST "${API_URL}" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d "{
    \"modelId\": {
      \"modelName\": \"${MODEL_NAME}\",
      \"modelVersion\": \"${MODEL_VERSION}\"
    },
    \"description\": \"Suburban LSTM model for predicting downlink traffic (PRB_DL) using 15-minute intervals. Input: 192 steps (2 days) → Output: 192 steps (2 days forecast)\",
    \"modelInformation\": {
      \"metadata\": {
        \"author\": \"AIMLFW User\",
        \"algorithm\": \"LSTM\",
        \"feature\": \"PRB_DL\",
        \"resampling_interval\": \"15 minutes\",
        \"input_steps\": \"192\",
        \"output_steps\": \"192\",
        \"forecast_horizon\": \"2 days\"
      },
      \"inputDataType\": \"PRB_DL\",
      \"outputDataType\": \"PRB_DL\"
    }
  }")

echo "Response:"
echo "$RESPONSE" | jq '.' 2>/dev/null || echo "$RESPONSE"

# Check if successful
if echo "$RESPONSE" | jq -e '.modelInfo // .modelId // .id // empty' > /dev/null 2>&1 || echo "$RESPONSE" | grep -q "already exists"; then
    echo ""
    echo "=========================================="
    echo "✅ Model registered successfully!"
    echo "=========================================="
    echo ""
    echo "Model: ${MODEL_NAME}/${MODEL_VERSION}"
    echo ""
    echo "You can now create a training job:"
    echo "  ./upload_and_train_suburban.sh BRANGES_T1"
else
    echo ""
    echo "⚠️  Model registration may have failed"
    echo "Please check the response above for errors"
    echo ""
    echo "If model already exists, that's okay - you can proceed with training job creation"
fi

