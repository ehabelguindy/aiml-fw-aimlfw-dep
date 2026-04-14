#!/bin/bash
# ==================================================================================
# Watch Training Live - Real-time Epoch-by-Epoch Monitoring
# ==================================================================================
# Shows ALL training output including every epoch, loss, MAE, validation metrics
# ==================================================================================

echo "=========================================="
echo "📺 LIVE TRAINING MONITOR"
echo "=========================================="
echo ""

# Get latest job ID if not provided
if [ -z "$1" ]; then
    if [ -f "automation_logs/latest_job_id.txt" ]; then
        TRAINING_JOB_ID=$(cat automation_logs/latest_job_id.txt)
        echo "📋 Using latest training job: $TRAINING_JOB_ID"
    else
        echo "Usage: $0 <training_job_id>"
        echo "   Or: $0  (will use latest job ID)"
        echo ""
        echo "Example: $0 150"
        exit 1
    fi
else
    TRAINING_JOB_ID=$1
fi

echo ""
echo "Job ID: $TRAINING_JOB_ID"
echo ""

# Wait for pod to start
echo "⏳ Waiting for training pod to start..."
for i in {1..15}; do
    # Find pod by job ID (runid label) or by pipeline name pattern
    POD=$(kubectl get pods -n kubeflow -l pipeline/runid=$TRAINING_JOB_ID --sort-by=.metadata.creationTimestamp 2>/dev/null | \
          grep -v NAME | tail -1 | awk '{print $1}')
    
    # If not found by label, try pattern matching
    if [ -z "$POD" ]; then
        POD=$(kubectl get pods -n kubeflow --sort-by=.metadata.creationTimestamp 2>/dev/null | \
              grep -E "suburban.*lstm.*traffic.*gpu.*v1[456].*system-container-impl" | \
              tail -1 | awk '{print $1}')
    fi
    STATUS=$(kubectl get pod "$POD" -n kubeflow -o jsonpath='{.status.phase}' 2>/dev/null 2>/dev/null || echo "NotFound")
    
    if [ "$STATUS" = "Running" ]; then
        echo "✅ Pod is running!"
        break
    elif [ "$STATUS" = "Error" ] || [ "$STATUS" = "Failed" ]; then
        echo "❌ Pod failed. Checking logs..."
        kubectl logs -n kubeflow "$POD" -c main --tail=50 2>&1 | tail -20
        exit 1
    fi
    
    echo "   Attempt $i/15: Status=$STATUS (pod may still be starting...)"
    sleep 1
done

if [ -z "$POD" ]; then
    echo "❌ Could not find training pod"
    echo ""
    echo "Available pods:"
    kubectl get pods -n kubeflow --sort-by=.metadata.creationTimestamp 2>/dev/null | tail -10
    exit 1
fi

echo ""
echo "📦 Training Pod: $POD"
echo ""

# Get pod status
POD_STATUS=$(kubectl get pod "$POD" -n kubeflow -o jsonpath='{.status.phase}' 2>/dev/null)
echo "Status: $POD_STATUS"
echo ""

# Show what we'll be monitoring
echo "=========================================="
echo "📊 LIVE TRAINING OUTPUT"
echo "=========================================="
echo ""
echo "You will see:"
echo "  • Every epoch progress (1/200, 2/200, etc.)"
echo "  • Training loss and MAE after each epoch"
echo "  • Validation loss and MAE after each epoch"
echo "  • Learning rate changes"
echo "  • Model checkpoint saves"
echo "  • Training completion"
echo ""
echo "Press Ctrl+C to stop monitoring (training will continue)"
echo ""
echo "=========================================="
echo ""

# Stream ALL logs in real-time (no filtering for full visibility)
kubectl logs -n kubeflow "$POD" -c main --follow 2>&1

echo ""
echo "=========================================="
echo "✅ Monitoring ended"
echo "=========================================="

