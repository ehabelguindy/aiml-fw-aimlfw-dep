#!/bin/bash
# ==================================================================================
# Create GPU-Enabled Training Job for LSTM Traffic Prediction
# ==================================================================================

TM_IP="10.237.129.136"
TM_PORT="32002"
API_URL="http://${TM_IP}:${TM_PORT}/ai-ml-model-training/v1/training-jobs"

echo "=========================================="
echo "Creating GPU-Enabled LSTM Training Job"
echo "=========================================="
echo ""

# Training job parameters
# Set default feature group
DEFAULT_FEATURE_GROUP="viavi"

FEATURE_GROUP="${1:-$DEFAULT_FEATURE_GROUP}"  # Use first argument or default
EPOCHS="${2:-150}"  # Default to 150 epochs if not specified

echo "ℹ️  Usage: $0 [feature_group] [epochs]"
echo "   Example: $0 my_feature_group 50"
echo "   Default: Using feature_group='$FEATURE_GROUP', epochs=$EPOCHS"
echo ""
MODEL_NAME="suburban-lstm-prbdl"  # S3-safe; must match MMS registration
MODEL_VERSION="1"
PIPELINE_NAME="Suburban_LSTM_Traffic_Prediction_Pipeline_GPU_V17"  # V17: Includes scaler_info.pkl in Model.zip

echo "Configuration:"
echo "  Model: ${MODEL_NAME}/${MODEL_VERSION}"
echo "  Feature Group: ${FEATURE_GROUP}"
echo "  Pipeline: ${PIPELINE_NAME}"
echo "  Epochs: ${EPOCHS}"
echo "  GPU: ENABLED (1 GPU per job)"
echo ""

# Create training job
echo "Creating GPU training job..."
RESPONSE=$(curl -s -X POST "${API_URL}" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d "{
    \"modelId\": {
      \"modelname\": \"${MODEL_NAME}\",
      \"modelversion\": \"${MODEL_VERSION}\"
    },
    \"model_location\": \"\",
    \"training_config\": {
      \"description\": \"GPU-accelerated LSTM model for predicting PRB_DL using 15-minute intervals\",
      \"dataPipeline\": {
        \"feature_group_name\": \"${FEATURE_GROUP}\",
        \"query_filter\": \"\",
        \"arguments\": {
          \"epochs\": \"${EPOCHS}\",
          \"modelname\": \"${MODEL_NAME}\",
          \"modelversion\": \"${MODEL_VERSION}\"
        }
      },
      \"trainingPipeline\": {
        \"training_pipeline_name\": \"${PIPELINE_NAME}\",
        \"training_pipeline_version\": \"${PIPELINE_NAME}\",
        \"retraining_pipeline_name\": \"${PIPELINE_NAME}\",
        \"retraining_pipeline_version\": \"${PIPELINE_NAME}\"
      }
    }
  }")

# Parse response
echo ""
echo "=========================================="
echo "Response:"
echo "=========================================="
echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"

# Extract training job ID
TRAINING_JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys, json; data = json.load(sys.stdin); print(data.get('id', ''))" 2>/dev/null)

if [ -n "$TRAINING_JOB_ID" ]; then
    echo ""
    echo "=========================================="
    echo "✅ GPU Training Job Created Successfully!"
    echo "=========================================="
    echo "Training Job ID: ${TRAINING_JOB_ID}"
    echo "Feature Group: ${FEATURE_GROUP}"
    echo "Model: ${MODEL_NAME}/${MODEL_VERSION}"
    echo "Epochs: ${EPOCHS}"
    echo "GPU: 1 x NVIDIA GPU"
    echo ""
    echo "📊 Monitor training:"
    echo "   Dashboard: http://localhost:30080"
    echo "   Training Manager: http://localhost:32002"
    echo ""
    echo "🔍 Check status:"
    echo "   kubectl get pods -n kubeflow | grep ${TRAINING_JOB_ID:0:20}"
    echo ""
    echo "📝 View logs:"
    echo "   kubectl logs -n kubeflow -l pipeline/runid=${TRAINING_JOB_ID} --tail=100 -f"
else
    echo ""
    echo "=========================================="
    echo "❌ Failed to Create Training Job"
    echo "=========================================="
    echo "Check the response above for error details"
    exit 1
fi

