#!/bin/bash
# ==================================================================================
# Setup System Crontab for LSTM Predictions
# ==================================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRON_LOG_DIR="$SCRIPT_DIR/automation_logs"
CRON_LOG_FILE="$CRON_LOG_DIR/cron.log"

# Create log directory
mkdir -p "$CRON_LOG_DIR"

# Get full paths to scripts
PREDICT_SCRIPT="$SCRIPT_DIR/predict.sh"
AUTOMATE_SCRIPT="$SCRIPT_DIR/automate_training_and_predict.sh"

# Check if automation script exists
if [ ! -f "$AUTOMATE_SCRIPT" ]; then
    echo "❌ ERROR: automate_training_and_predict.sh not found at $AUTOMATE_SCRIPT"
    exit 1
fi

# Make sure scripts are executable
chmod +x "$AUTOMATE_SCRIPT"
chmod +x "$PREDICT_SCRIPT"

# Create cron entry - Run every 15 minutes
CRON_SCHEDULE="*/15 * * * *"
CRON_COMMAND="cd $SCRIPT_DIR && bash $AUTOMATE_SCRIPT >> $CRON_LOG_FILE 2>&1"

# Check if automation process is currently running
echo "Checking for running automation processes..."
RUNNING_PROCESSES=$(ps aux | grep -c "[a]utomate_training_and_predict" || echo "0")
if [ "$RUNNING_PROCESSES" -gt 0 ]; then
    echo "⚠️  WARNING: Automation process is currently running!"
    echo ""
    echo "Running processes:"
    ps aux | grep "[a]utomate_training_and_predict" | grep -v grep
    echo ""
    read -p "Do you want to continue anyway? This may cause duplicate runs. (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled. Please wait for current automation to complete."
        exit 1
    fi
    echo "⚠️  Continuing despite running processes..."
    echo ""
fi

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "$AUTOMATE_SCRIPT\|automate_training"; then
    echo "⚠️  Cron job already exists for automation"
    echo ""
    echo "Current crontab entries:"
    crontab -l 2>/dev/null | grep -E "automate_training|predict|LSTM" || echo "  (none found)"
    echo ""
    read -p "Do you want to replace it? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
    # Remove old entry
    crontab -l 2>/dev/null | grep -v "$AUTOMATE_SCRIPT" | grep -v "automate_training" | crontab -
fi

# Add new cron job
(crontab -l 2>/dev/null; echo "$CRON_SCHEDULE $CRON_COMMAND") | crontab -

echo "✅ Cron job added successfully!"
echo ""
echo "Schedule: $CRON_SCHEDULE (every 15 minutes)"
echo "Script: $AUTOMATE_SCRIPT"
echo "Logs: $CRON_LOG_FILE"
echo ""
echo "🚀 Running job immediately..."
bash "$AUTOMATE_SCRIPT" >> "$CRON_LOG_FILE" 2>&1 &
AUTOMATE_PID=$!
echo "✅ Job started in background (PID: $AUTOMATE_PID)"
echo "   (Check logs with: tail -f $CRON_LOG_FILE)"
echo "   (Check status with: ./monitor_automation.sh)"
echo ""
echo "Current crontab:"
crontab -l | grep -E "automate|predict|LSTM" || echo "  (check with: crontab -l)"
echo ""
echo "To view logs: tail -f $CRON_LOG_FILE"
echo "To monitor: ./monitor_automation.sh"
echo "To remove: ./stop_automation.sh"

