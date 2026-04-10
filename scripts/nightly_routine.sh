#!/bin/bash
# Daedalus Automated Nightly Routine (Sleep -> Dream -> Wake)

echo "=========================================================="
echo "Starting Daedalus Nightly Routine at $(date)"
echo "=========================================================="

DAEDALUS_DIR="/mnt/projects1/daedalus"
PYTHON_BIN="/home/infinity-forecast/anaconda3/envs/daedalus-env/bin/python"

cd $DAEDALUS_DIR || exit 1

# 1. SLEEP PHASE
echo "[1/3] SLEEP PHASE: Suspending active API Server to free VRAM..."
# Send graceful SIGTERM to the api server
pkill -TERM -f "api_server.py"
# Wait briefly for VRAM to clear
sleep 5

# 2. DREAM PHASE
echo "[2/3] DREAM PHASE: Running night cycle (consolidation and fine-tuning)..."
$PYTHON_BIN scripts/run_night_cycle.py
NIGHT_EXIT_CODE=$?

if [ $NIGHT_EXIT_CODE -ne 0 ]; then
    echo "WARNING: Night cycle finished with errors (exit code $NIGHT_EXIT_CODE)."
else
    echo "Night cycle completed successfully."
fi

# 3. WAKE PHASE
echo "[3/3] AWAKENING: Restarting API Server..."
# Use nohup to detach it from this script's session so Daedalus is ready all day
nohup $PYTHON_BIN scripts/api_server.py --host 0.0.0.0 >> logs/api_server.log 2>&1 &

echo "=========================================================="
echo "Routine completed at $(date)."
echo "=========================================================="
