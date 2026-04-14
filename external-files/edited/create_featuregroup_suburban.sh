#!/bin/bash
# ==================================================================================
# Create FeatureGroup for Suburban Dataset
# ==================================================================================

TM_IP="localhost"
TM_PORT="32002"

FEATURE_GROUP_NAME="suburban"
BUCKET_NAME="Suburban_Dataset"
MEASUREMENT="cell_metrics"
ENTITY_COLUMN="CELLULE"
FEATURE_COLUMN="PRB_DL"

echo "=========================================="
echo "Creating FeatureGroup for Suburban Dataset"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  FeatureGroup Name: ${FEATURE_GROUP_NAME}"
echo "  InfluxDB Bucket: ${BUCKET_NAME}"
echo "  Measurement: ${MEASUREMENT}"
echo "  Entity Column: ${ENTITY_COLUMN}"
echo "  Feature Column: ${FEATURE_COLUMN}"
echo ""

# Create FeatureGroup
echo "Creating FeatureGroup..."
# Use the correct API endpoint: /ai-ml-model-training/v1/featureGroup
RESPONSE=$(curl -s -X POST "http://${TM_IP}:${TM_PORT}/ai-ml-model-training/v1/featureGroup" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d "{
  \"featuregroup_name\": \"${FEATURE_GROUP_NAME}\",
  \"feature_list\": \"${FEATURE_COLUMN}\",
  \"datalake_source\": \"InfluxSource\",
  \"enable_dme\": false,
  \"host\": \"my-release-influxdb.default.svc.cluster.local\",
  \"port\": \"8086\",
  \"dme_port\": \"\",
  \"bucket\": \"${BUCKET_NAME}\",
  \"token\": \"Dl8FVVIaYMUhJoG1kSmZ\",
  \"source_name\": \"\",
  \"measured_obj_class\": \"\",
  \"measurement\": \"${MEASUREMENT}\",
  \"db_org\": \"primary\"
}")

echo "Response:"
echo "$RESPONSE" | jq '.' 2>/dev/null || echo "$RESPONSE"

# Check if successful (response might be empty or contain success message)
if echo "$RESPONSE" | jq -e '.featuregroup_name // .FeatureGroup // .message // empty' > /dev/null 2>&1 || [ -z "$RESPONSE" ]; then
    echo ""
    echo "=========================================="
    echo "✅ FeatureGroup created successfully!"
    echo "=========================================="
    echo ""
    echo "You can now use this FeatureGroup in training jobs:"
    echo "  FeatureGroup Name: ${FEATURE_GROUP_NAME}"
    echo ""
    echo "View in dashboard:"
    echo "  http://localhost:32005/FeatureGroup/ListFeatureGroups"
else
    echo ""
    echo "⚠️  FeatureGroup creation may have failed"
    echo "Please check the response above for errors"
    echo ""
    echo "You can also create it manually via the dashboard:"
    echo "  http://localhost:32005/FeatureGroup/CreateFeatureGroup"
fi

