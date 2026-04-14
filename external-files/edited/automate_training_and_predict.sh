#!/bin/bash
# ==================================================================================
# Automated Training and Prediction Workflow
# ==================================================================================
# This script:
#   1. Checks if new data arrived in InfluxDB
#   2. Creates training job and saves weights
#   3. Sends to KServe to predict
#   4. Saves all steps in log file
# ==================================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
FEATURE_GROUP="${FEATURE_GROUP:-suburban}"
EPOCHS="${EPOCHS:-200}"
CELL_NAME="${CELL_NAME:-BRANGES_T1}"
AUTO_TRAIN="${AUTO_TRAIN:-true}"  # Set to "false" to skip training, only predict

# Logging setup
LOG_DIR="./automation_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/automation_${TIMESTAMP}.log"
STATUS_FILE="$LOG_DIR/status.txt"
CONSOLIDATED_LOG="$LOG_DIR/all_automation_logs_consolidated.log"

# Initialize consolidated log with separator if this is the start of a new run
if [ ! -f "$CONSOLIDATED_LOG" ]; then
    touch "$CONSOLIDATED_LOG"
fi

# Function to log messages (writes to both individual log file and consolidated log)
log() {
    local message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    # Write to individual log file
    echo "$message" | tee -a "$LOG_FILE"
    # Also append to consolidated log file
    echo "$message" >> "$CONSOLIDATED_LOG"
    # Also update status file
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" > "$STATUS_FILE"
}

# Add separator to consolidated log for this run
{
    echo ""
    echo "============================================================"
    echo "LOG FILE: automation_${TIMESTAMP}.log"
    echo "============================================================"
    echo ""
} >> "$CONSOLIDATED_LOG"

log "=========================================="
log "AUTOMATED TRAINING AND PREDICTION"
log "=========================================="
log "Feature Group: $FEATURE_GROUP"
log "Epochs: $EPOCHS"
log "Cell: $CELL_NAME"
log "Auto Train: $AUTO_TRAIN"
log "Log File: $LOG_FILE"
log ""

# ==================================================================================
# STEP 0: Ensure InfluxDB Port-Forward is Active
# ==================================================================================
log "STEP 0: Ensuring InfluxDB port-forward is active..."
if ! lsof -i:8086 >/dev/null 2>&1; then
    log "   Setting up InfluxDB port-forward..."
    kubectl port-forward -n default svc/my-release-influxdb 8086:8086 > /tmp/influxdb_portforward.log 2>&1 &
    sleep 5
    if lsof -i:8086 >/dev/null 2>&1; then
        log "   ✅ InfluxDB port-forward active"
    else
        log "   ⚠️  Port-forward setup failed, but continuing..."
    fi
else
    log "   ✅ InfluxDB port-forward already active"
fi
log ""

# ==================================================================================
# STEP 1: Check for New Data in InfluxDB
# ==================================================================================
log "STEP 1: Checking for new data in InfluxDB..."
log "Status: Checking..."

if python3 check_influxdb_new_data.py 2>&1 | tee -a "$LOG_FILE" "$CONSOLIDATED_LOG"; then
    HAS_NEW_DATA=true
    log "✅ New data found in InfluxDB"
    log "Status: New data found"
else
    HAS_NEW_DATA=false
    log "ℹ️  No new data in InfluxDB"
    log "Status: No new data"
fi
log ""

# ==================================================================================
# STEP 2: Create Training Job (always, after checking InfluxDB)
# ==================================================================================
if [ "$AUTO_TRAIN" = "true" ]; then
    log "STEP 2: Creating training job..."
    log "Status: Creating training job..."
    
    if [ "$HAS_NEW_DATA" = "true" ]; then
        log "   New data found - creating training job"
    else
        log "   No new data found - creating training job anyway (as requested)"
    fi
    
    # Create training job
    if bash create_training_job_gpu.sh "$FEATURE_GROUP" "$EPOCHS" 2>&1 | tee -a "$LOG_FILE" "$CONSOLIDATED_LOG"; then
        # Extract training job ID from log
        TRAINING_JOB_ID=$(grep "Training Job ID:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
        
        if [ -n "$TRAINING_JOB_ID" ]; then
            log "✅ Training job created: $TRAINING_JOB_ID"
            log "Status: Training job $TRAINING_JOB_ID created"
            
            # Save job ID to file for monitoring
            JOB_ID_FILE="$LOG_DIR/latest_job_id.txt"
            echo "$TRAINING_JOB_ID" > "$JOB_ID_FILE"
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Job ID: $TRAINING_JOB_ID" >> "$LOG_DIR/job_history.txt"
            
            # Wait for training to complete (optional - you can add monitoring here)
            log "⏳ Training job started. Model will be saved automatically when complete."
            log "   (You can monitor: kubectl get pods -n kubeflow | grep training)"
            
            # Clean up old training jobs (keep only last 3)
            log "🧹 Cleaning up old training jobs (keeping last 3)..."
            bash "$SCRIPT_DIR/cleanup_old_training_jobs.sh" 2>&1 | tee -a "$LOG_FILE" "$CONSOLIDATED_LOG" || true
        else
            log "⚠️  Training job created but ID not found in response"
            log "Status: Training job created (ID unknown)"
        fi
    else
        log "❌ Failed to create training job"
        log "Status: Training job failed"
        exit 1
    fi
    log ""
else
    log "STEP 2: Skipped (AUTO_TRAIN=false)"
    log "Status: Training skipped"
    log ""
fi

# ==================================================================================
# STEP 3: Send to KServe to Predict
# ==================================================================================
log "STEP 3: Running predictions via KServe..."
log "Status: Running predictions..."

if bash predict.sh 2>&1 | tee -a "$LOG_FILE" "$CONSOLIDATED_LOG"; then
    log "✅ Predictions completed successfully"
    log "Status: Predictions completed"
    
    # Find latest prediction file
    LATEST_PRED=$(ls -t ./predictions/prediction_*.json 2>/dev/null | head -1)
    if [ -n "$LATEST_PRED" ]; then
        log "   Prediction saved to: $LATEST_PRED"
    fi
else
    log "❌ Predictions failed"
    log "Status: Predictions failed"
    exit 1
fi
log ""

# ==================================================================================
# STEP 4: Save Summary
# ==================================================================================
log "STEP 4: Saving summary..."
log "Status: Completed"

# Create summary
SUMMARY_FILE="$LOG_DIR/summary_${TIMESTAMP}.txt"
{
    echo "=========================================="
    echo "AUTOMATION SUMMARY"
    echo "=========================================="
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Feature Group: $FEATURE_GROUP"
    echo "Epochs: $EPOCHS"
    echo "Cell: $CELL_NAME"
    echo ""
    echo "Step 1 - New Data Check:"
    echo "  Result: $([ "$HAS_NEW_DATA" = "true" ] && echo "New data found" || echo "No new data")"
    echo ""
    if [ "$AUTO_TRAIN" = "true" ]; then
        echo "Step 2 - Training Job:"
        echo "  Status: Created"
        [ -n "$TRAINING_JOB_ID" ] && echo "  Job ID: $TRAINING_JOB_ID"
        echo "  Note: Created regardless of new data status"
        echo ""
    fi
    echo "Step 3 - Predictions:"
    echo "  Status: Completed"
    [ -n "$LATEST_PRED" ] && echo "  File: $LATEST_PRED"
    echo ""
    echo "Full log: $LOG_FILE"
    echo "=========================================="
} > "$SUMMARY_FILE"

log "✅ Summary saved to: $SUMMARY_FILE"
log ""
log "=========================================="
log "AUTOMATION COMPLETE"
log "=========================================="
log "Full log: $LOG_FILE"
log "Summary: $SUMMARY_FILE"
log "Status: $STATUS_FILE"

# Add end separator to consolidated log for this run
{
    echo ""
    echo "============================================================"
    echo "END OF: automation_${TIMESTAMP}.log"
    echo "============================================================"
    echo ""
} >> "$CONSOLIDATED_LOG"

