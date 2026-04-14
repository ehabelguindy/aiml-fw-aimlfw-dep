# ==================================================================================
#
#       LSTM Traffic Prediction Pipeline for Suburban Dataset
#       Uses PRB_DL column from feature group with 15-minute intervals
#       Input: 192 steps (2 days) → Output: 192 steps (2 days forecast)
#
# ==================================================================================

#!/usr/bin/env python
# coding: utf-8

import kfp
import kfp.dsl as dsl
from kfp.dsl import InputPath, OutputPath
from kfp.dsl import component as component
from kfp.dsl import ContainerSpec
import requests

# Try to import kubernetes for setting imagePullPolicy
try:
    from kfp import kubernetes
except ImportError:
    try:
        from kfp.dsl import kubernetes
    except ImportError:
        kubernetes = None

BASE_IMAGE = "traininghost/pipelinegpuimage:latest"

# step 1: train_export_model component
# This component trains the LSTM model and exports it as a TensorFlow SavedModel.
# It uses the featurepath, epochs, modelname, and modelversion parameters to train the model.
# It also uses the pod_spec_patch parameter to specify the Kubernetes pod settings for the training task.
# The pod_spec_patch parameter is a JSON string that contains the Kubernetes pod settings for the training task.
# The pod_spec_patch parameter is used by KFP, not needed in function body.


@component(base_image=BASE_IMAGE)
def train_export_model(featurepath: str, epochs: str, modelname: str, modelversion: str, pod_spec_patch: str = ""):
    """
    Train LSTM model to predict downlink traffic using PRB_DL from Suburban Dataset.
    Data is at 15-minute intervals.
    Input: 192 steps (2 days) → Output: 192 steps (2 days forecast)
    
    Args:
        featurepath: Feature group name
        epochs: Number of training epochs
        modelname: Name of the model
        modelversion: Version of the model
        pod_spec_patch: JSON string for pod spec patch (used by KFP, not needed in function body)
    """
    
    # Import all required modules
    import tensorflow as tf
    from numpy import array
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    import numpy as np
    import requests
    import pandas as pd
    import os
    from datetime import datetime, timedelta
    from featurestoresdk.feature_store_sdk import FeatureStoreSdk
    from modelmetricsdk.model_metrics_sdk import ModelMetricsSdk
    
    print("### CODE VERSION: V17-SCALER-FIX ###", flush=True)
    print("=" * 60)
    print("LSTM Suburban Traffic Prediction Pipeline")
    print("=" * 60)
    print(f"Feature path: {featurepath}")
    print(f"Model name: {modelname}, Version: {modelversion}")
    print(f"Epochs: {epochs}")
    print("Configuration: 192 input steps → 192 output steps (2 days @ 15min intervals)")
    print("=" * 60)
    
    # Initialize SDKs
    fs_sdk = FeatureStoreSdk()
    mm_sdk = ModelMetricsSdk()
    
    # Step 1: Extract data features using FeatureStoreSdk
    # Using PRB_DL from Suburban Dataset (matches feature group feature_list)
    print("\n[Step 1] Extracting features from Feature Store...")
    print(f"Requesting feature: PRB_DL")
    
    try:
        features = fs_sdk.get_features(featurepath, ['PRB_DL'])
        print("[OK] Found PRB_DL column")
    except Exception as e:
        print(f"[ERROR] Could not find PRB_DL column")
        print(f"Error: {e}")
        raise
    
    print(f"\nDataFrame shape: {features.shape}")
    print(f"Columns: {features.columns.tolist()}")
    print("\nFirst few rows:")
    print(features.head())
    
    # Step 2: Process data (already at 15-minute intervals, no resampling needed)
    print("\n[Step 2] Processing data (15-minute intervals, no resampling needed)...")
    
    # Ensure timestamp column exists and is in datetime format
    if 'timestamp' in features.columns:
        features['timestamp'] = pd.to_datetime(features['timestamp'], unit='s', errors='coerce')
    elif 'time' in features.columns:
        features['timestamp'] = pd.to_datetime(features['time'], unit='s', errors='coerce')
    else:
        # If no timestamp column, create one from index
        print("Warning: No timestamp column found, using index as timestamp")
        features['timestamp'] = pd.date_range(start='2024-01-01', periods=len(features), freq='15min')
    
    # Set timestamp as index
    features_indexed = features.set_index('timestamp')
    
    # Convert PRB_DL to numeric (already named correctly)
    # Note: Feature Store SDK returns column names as-is
    features_indexed['PRB_DL'] = pd.to_numeric(features_indexed['PRB_DL'], errors='coerce')
    
    # Remove any NaN values
    features_indexed = features_indexed.dropna(subset=['PRB_DL'])
    
    # Sort by timestamp to ensure chronological order
    features_indexed = features_indexed.sort_index()
    
    print(f"Total data points (15-min intervals): {len(features_indexed)}")
    print(f"Date range: {features_indexed.index.min()} to {features_indexed.index.max()}")
    print(f"\nData sample:")
    print(features_indexed.head(10))
    
    # Extract PRB_DL values
    prb_dl_values = features_indexed['PRB_DL'].values
    
    # Step 3: Normalize data using manual Min-Max normalization
    print("\n[Step 3] Normalizing data...")
    min_val = float(prb_dl_values.min())
    max_val = float(prb_dl_values.max())
    range_val = max_val - min_val if max_val != min_val else 1.0
    prb_dl_scaled = (prb_dl_values - min_val) / range_val
    
    print(f"Original PRB_DL range: [{min_val:.2f}, {max_val:.2f}]")
    print(f"Scaled PRB_DL range: [{prb_dl_scaled.min():.2f}, {prb_dl_scaled.max():.2f}]")
    
    # Step 4: Prepare data for LSTM (sequence-to-sequence prediction)
    # 192 input steps (2 days) → 192 output steps (2 days forecast)
    print("\n[Step 4] Preparing sequences for LSTM model...")
    print("Creating sequences: 192 input steps → 192 output steps")
    
    def split_series(series, n_past, n_future):
        """
        Split time series into sequences for LSTM.
        n_past: number of past time steps to use as input (192)
        n_future: number of future time steps to predict (192)
        """
        X, y = list(), list()
        for window_start in range(len(series)):
            past_end = window_start + n_past
            future_end = past_end + n_future
            if future_end > len(series):
                break
            # Slicing the past and future parts of the window
            past = series[window_start:past_end]
            future = series[past_end:future_end]
            X.append(past)
            y.append(future)
        return np.array(X), np.array(y)
    
    # 192 steps = 2 days * 24 hours * 4 intervals per hour
    n_past = 192   # Input: 2 days of history
    n_future = 192 # Output: 2 days of forecast
    
    X, y = split_series(prb_dl_scaled, n_past, n_future)
    
    # Reshape for LSTM: (samples, time_steps, features)
    # Since we're using a single feature (PRB_DL), features=1
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape((y.shape[0], y.shape[1]))
    
    print(f"Input shape (X): {X.shape}")
    print(f"Output shape (y): {y.shape}")
    print(f"Training samples: {X.shape[0]}")
    print(f"Each sample: {n_past} input steps → {n_future} output steps")
    
    # Step 5: Build LSTM model for sequence-to-sequence prediction
    print("\n[Step 5] Building LSTM model for 192→192 prediction...")
    ########################

    # optional steps
    # ====== GPU OPTIMIZATION: Enable Mixed Precision Training ======
    # This gives 2-3x speedup on modern GPUs (RTX 5000, T4, etc.)
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print(f"[OK] Mixed Precision enabled: {policy.name} (2-3x faster on GPU)")
    
    # ====== GPU OPTIMIZATION: Multi-GPU Training with MirroredStrategy ======
    # Use all available GPUs for 2x speedup (if 2 GPUs available)
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    print(f"[INFO] Found {len(gpus)} GPU(s)")
    
    # ====== GPU OPTIMIZATION: Pin CPU Threading (reduces thread scheduling overhead) ======
    # On multi-GPU nodes, CPU thread scheduling can starve the GPU
    # Setting explicit thread counts prevents CPU bottleneck
    tf.config.threading.set_inter_op_parallelism_threads(4)   # Parallel operations between ops
    tf.config.threading.set_intra_op_parallelism_threads(8)   # Parallel operations within an op
    print(f"[OK] CPU threading optimized: inter_op=4, intra_op=8 (reduces CPU-GPU coordination overhead)")
    
    if len(gpus) > 1:
        # Use MirroredStrategy for multi-GPU training
        strategy = tf.distribute.MirroredStrategy()
        print(f"[OK] Multi-GPU training enabled: {strategy.num_replicas_in_sync} GPU(s) - ~{strategy.num_replicas_in_sync}x speedup")
    else:
        # Single GPU - use default strategy
        strategy = tf.distribute.get_strategy()
        print(f"[INFO] Single GPU training (using default strategy)")
    
    # ====== GPU OPTIMIZATION: XLA DISABLED (incompatible with CUDA graphs + validation split) ======
    # Note: XLA caused CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE with tf.data validation
    # Mixed Precision + Multi-GPU provides significant speedup
    print("[INFO] XLA JIT compilation DISABLED (incompatible with this configuration)")
    ##################################################

    # N = number of input time steps (192)
    # H = number of output time steps (192)
    N = X.shape[1]  # Input sequence length
    H = n_future     # Output sequence length
    
    # Build model within strategy scope for multi-GPU support
    with strategy.scope():
        regressor = Sequential()
        
        # First LSTM layer: 200 units, return_sequences=True, input_shape=(N, 1)
        # Default activation: tanh (for LSTM gates) and sigmoid (for forget/input/output gates)
        regressor.add(LSTM(200, return_sequences=True, input_shape=(N, 1)))
        regressor.add(Dropout(0.15))  # Reduced dropout from 0.2 to 0.15 to allow more learning capacity
        
        # Second LSTM layer: 200 units, return_sequences=True
        regressor.add(LSTM(200, return_sequences=True))
        regressor.add(Dropout(0.15))  # Reduced dropout from 0.2 to 0.15 to allow more learning capacity
        
        # Third LSTM layer: 200 units, return_sequences=False (final LSTM layer)
        regressor.add(LSTM(200, return_sequences=False))
        regressor.add(Dropout(0.15))  # Reduced dropout from 0.2 to 0.15 to allow more learning capacity
        
        # Dense output layer: H units (linear output for 192-step forecast)
        # Use float32 for final layer (required for mixed precision)
        regressor.add(Dense(H, dtype='float32'))
        
        # Use the regressor as the model
        model = regressor
        
        # Compile model with Adam optimizer and learning rate 0.002
        from tensorflow.keras.optimizers import Adam
        optimizer = Adam(learning_rate=0.002)
        # Optimize compilation for speed: steps_per_execution batches multiple steps, fewer metrics
        model.compile(
            loss='mse', 
            optimizer=optimizer, 
            metrics=['mae'],  # Only track MAE (MSE redundant with loss)
            steps_per_execution=20,  # Increased to 20 for better GPU utilization
            run_eagerly=False  # Ensure graph mode execution (faster)
        )
    
    print("\nModel Architecture:")
    model.summary()
    
    # Step 6: Train the model with Callbacks
    print("\n[Step 6] Training LSTM model with callbacks...")
    
    # ====== GPU OPTIMIZATION: Optimal batch size for RTX 5000 with Mixed Precision + Multi-GPU ======
    # Increased batch size for better GPU utilization (only using 3% memory currently)
    # With multi-GPU, effective batch size = batch_size_per_replica * num_gpus
    # Note: LSTMs are less parallelizable than CNNs, so multi-GPU speedup is limited (~1.3-1.5x vs 2x)
    batch_size_per_replica = 256  # Increased from 128 to maximize GPU utilization
    num_replicas = strategy.num_replicas_in_sync
    effective_batch_size = batch_size_per_replica * num_replicas
    training_epochs = int(epochs)
    
    print(f"Training Configuration:")
    print(f"  Max Epochs: {training_epochs}")
    print(f"  Batch Size per GPU: {batch_size_per_replica}")
    print(f"  Effective Batch Size: {effective_batch_size} ({num_replicas} GPU(s))")
    print(f"  Learning Rate: 0.002 (with ReduceLROnPlateau)")
    print(f"  Mixed Precision: Enabled (float16) - 2-3x faster")
    print(f"  Multi-GPU: {num_replicas} GPU(s) - ~{num_replicas}x speedup")
    print(f"  CPU Threading: inter_op=4, intra_op=8 (reduces CPU-GPU coordination overhead)")
    print(f"  XLA Compilation: Disabled (compatibility issue)")
    print(f"Note: Multi-GPU + Mixed Precision + CPU Threading provides significant speedup")
    
   
    # ====== GPU OPTIMIZATION: Use tf.data for efficient data loading ======
    # Split data manually for better control
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Create tf.data datasets with aggressive optimizations for speed
    # Cache data in memory after first epoch to avoid repeated preprocessing
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.cache()  # Cache in memory after first epoch
    train_dataset = train_dataset.shuffle(buffer_size=min(4096, len(X_train)), reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(batch_size_per_replica, drop_remainder=True)  # drop_remainder for better performance
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)  # Prefetch multiple batches
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.cache()  # Cache validation data
    # Use drop_remainder=False for validation to allow partial batches (important for small validation sets)
    val_dataset = val_dataset.batch(batch_size_per_replica, drop_remainder=False)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    
    print(f"[OK] Data pipeline optimized: caching + prefetching (faster after first epoch)")
    
    # ====== CALLBACKS: Model Checkpoint, LR Scheduler ======
    from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, LambdaCallback
    
    # Create training_logs directory and log file for detailed epoch-by-epoch logging
    # Log file will be saved to host training_logs folder (not in Model.zip)
    # Try volume mount path first, fall back to relative path
    volume_mount_path = '/workspace/training_logs'
    if os.path.exists(volume_mount_path):
        training_logs_dir = volume_mount_path
    else:
        training_logs_dir = './training_logs'
    os.makedirs(training_logs_dir, exist_ok=True)
    from datetime import datetime
    log_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    epoch_log_file = f'{training_logs_dir}/epochs_{log_timestamp}.log'
    
    # Initialize log file with header (will write training_epochs after it's defined)
    with open(epoch_log_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("EPOCH TRAINING LOG\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {modelname} v{modelversion}\n")
        f.write(f"Feature Path: {featurepath}\n")
        f.write(f"Training Epochs: {epochs}\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
    ###################################################
    # Custom callback to print and log detailed metrics after each epoch
    import sys
    def print_epoch_metrics(epoch, logs):
        """Print formatted metrics summary after each epoch and write to log file"""
        # Get metrics (handle missing keys gracefully)
        train_loss = logs.get('loss', 0)
        train_mae = logs.get('mae', 0)
        val_loss = logs.get('val_loss', 0)
        val_mae = logs.get('val_mae', 0)
        lr = logs.get('lr', 0)
        
        epoch_num = epoch + 1
        
        # Print formatted summary with flush to ensure immediate output
        print(f"\n{'='*60}", flush=True)
        print(f"[Epoch {epoch_num} Summary]", flush=True)
        print(f"{'='*60}", flush=True)
        print(f"  Training   - Loss: {train_loss:.6f}, MAE: {train_mae:.6f}", flush=True)
        print(f"  Validation - Loss: {val_loss:.6f}, MAE: {val_mae:.6f}", flush=True)
        print(f"  Learning Rate: {lr:.6f}", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        # Write to log file
        with open(epoch_log_file, 'a') as f:
            f.write(f"Epoch {epoch_num}/{training_epochs}\n")
            f.write(f"  Training Loss: {train_loss:.6f}\n")
            f.write(f"  Training MAE:  {train_mae:.6f}\n")
            f.write(f"  Val Loss:      {val_loss:.6f}\n")
            f.write(f"  Val MAE:       {val_mae:.6f}\n")
            f.write(f"  Learning Rate: {lr:.6f}\n")
            f.write("-" * 60 + "\n")
    # Prints and writes metrics after each epoch. It's a callback function that is called after each epoch.
    metrics_callback = LambdaCallback(on_epoch_end=print_epoch_metrics)
    
    print(f"[OK] Epoch log file: {epoch_log_file}")
    #########################################################################################

    # Reduces learning rate if validation loss stops improving. It's a callback function that is called after each epoch.
    # Learning Rate Scheduler: Reduce LR when validation loss plateaus
     # ====== CALLBACKS: Model Checkpoint, LR Scheduler ======
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,        # Reduce LR by half
        patience=20,       # Increased from 15 to 20 epochs before reducing LR (less aggressive)
        min_lr=0.00001,    # Minimum learning rate
        verbose=1,         # Show LR reduction messages (minimal speed impact)
        mode='min'
    )
    # Saves the best model seen during training.
    # Model Checkpoint: Save best model during training
    checkpoint = ModelCheckpoint(
        filepath='./best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1,         # Show checkpoint save messages (minimal speed impact)
        mode='min'
    )
    
    print(f"[OK] Callbacks configured:")
    print(f"   - Metrics Summary (prints detailed metrics after each epoch)")
    print(f"   - ReduceLROnPlateau (patience=20, factor=0.5)")
    print(f"   - ModelCheckpoint (saves best model)")
    
    # Write training configuration to log file
    with open(epoch_log_file, 'a') as f:
        f.write("\nTraining Configuration:\n")
        f.write(f"  Max Epochs: {training_epochs}\n")
        f.write(f"  Batch Size per GPU: {batch_size_per_replica}\n")
        f.write(f"  Effective Batch Size: {effective_batch_size} ({num_replicas} GPU(s))\n")
        f.write(f"  Learning Rate: 0.002 (with ReduceLROnPlateau)\n")
        f.write(f"  Mixed Precision: Enabled (float16)\n")
        f.write(f"  Multi-GPU: {num_replicas} GPU(s)\n")
        f.write(f"  CPU Threading: inter_op=4, intra_op=8\n")
        f.write(f"  Training Samples: {len(X_train)}\n")
        f.write(f"  Validation Samples: {len(X_val)}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("EPOCH PROGRESS\n")
        f.write("=" * 60 + "\n\n")
    
   
    # Create a Tee class to write to both stdout and log file (capture raw training output)
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    # Redirect stdout/stderr to both console and log file to capture raw training output
    log_file_handle = open(epoch_log_file, 'a')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    try:
        # Create Tee that writes to both stdout and log file
        sys.stdout = Tee(original_stdout, log_file_handle)
        sys.stderr = Tee(original_stderr, log_file_handle)
        
        # Train with optimized data pipeline and callbacks
        # Optimized for speed: cached datasets, steps_per_execution, verbose callbacks for monitoring
        # All output (including progress bars) will be captured in the log file
        history = model.fit(
            train_dataset, 
            epochs=training_epochs, 
            validation_data=val_dataset, 
            callbacks=[metrics_callback, reduce_lr, checkpoint],
            verbose=1  # Show progress bar with metrics for each epoch (captured to log file)
        )
    finally:
        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file_handle.close()
    
    # Optional steps: Save training history in different formats to be used for analysis and visualization
    #Save training history in different formats to be used for analysis and visualization
    # ====== SAVE EPOCH LOGS ======
    print("\n[Saving Epoch-by-Epoch Training Logs]", flush=True)
    import json
    import os, traceback
    
    # Use existing training_logs_dir (already created above)
    # Use same timestamp as epoch log file
    timestamp = log_timestamp
    
    # Finalize epoch log file
    with open(epoch_log_file, 'a') as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write("TRAINING COMPLETED\n")
        f.write("=" * 60 + "\n")
        f.write(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Epochs: {len(history.history['loss'])}\n")
        f.write("=" * 60 + "\n")
    
    # Write check results directly into the log file (so we can see them even if stdout is hidden)
    try:
        abs_path = os.path.abspath(epoch_log_file)
        exists = os.path.exists(epoch_log_file)
        size = os.path.getsize(epoch_log_file) if exists else -1
        
        with open(epoch_log_file, "a") as f:
            f.write("\n[POST-CHECK]\n")
            f.write(f"exists={exists}\n")
            f.write(f"abs_path={abs_path}\n")
            f.write(f"cwd={os.getcwd()}\n")
            if os.path.exists(training_logs_dir):
                f.write(f"dir_contents={os.listdir(training_logs_dir)}\n")
            f.write(f"size_bytes={size}\n")
        
        # Also print to stdout (with flush)
        print("\n[DEBUG] Starting epoch log file verification...", flush=True)
        abs_epoch_log_file = os.path.abspath(epoch_log_file)
        print(f"[DEBUG] epoch_log_file = {epoch_log_file}", flush=True)
        print(f"[DEBUG] abs path      = {abs_epoch_log_file}", flush=True)
        print(f"[DEBUG] cwd           = {os.getcwd()}", flush=True)
        
        if os.path.exists(training_logs_dir):
            print(f"[DEBUG] training_logs_dir exists: {training_logs_dir}", flush=True)
            print(f"[DEBUG] contents: {os.listdir(training_logs_dir)}", flush=True)
        else:
            print(f"[DEBUG] training_logs_dir MISSING: {training_logs_dir}", flush=True)
        
        if exists:
            print(f"[OK] Epoch log file completed: {epoch_log_file}", flush=True)
            print(f"[OK] Log file size: {size} bytes", flush=True)
        else:
            print(f"[WARNING] Epoch log file NOT found: {epoch_log_file}", flush=True)
    
    except Exception as e:
        # Write error to log file
        try:
            with open(epoch_log_file, "a") as f:
                f.write("\n[POST-CHECK-ERROR]\n")
                f.write(str(e) + "\n")
                f.write(traceback.format_exc() + "\n")
        except:
            pass  # If we can't write to the log file, at least try stdout
        print(f"[ERROR] Exception while checking epoch log file: {e}", flush=True)
        traceback.print_exc()
    
    # Extract history (only MAE is tracked, not MSE - matches model.compile metrics)
    epoch_logs = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'mae': [float(x) for x in history.history['mae']],
        'val_mae': [float(x) for x in history.history['val_mae']],
        'epochs_trained': len(history.history['loss']),
        'early_stopped': len(history.history['loss']) < training_epochs,
        'best_val_loss': float(min(history.history['val_loss'])),
        'best_val_loss_epoch': int(history.history['val_loss'].index(min(history.history['val_loss'])) + 1),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'final_train_mae': float(history.history['mae'][-1]),
        'final_val_mae': float(history.history['val_mae'][-1]),
        'model_name': modelname,
        'model_version': modelversion,
        'feature_path': featurepath,
        'training_started': timestamp
    }
    
    # Save to JSON file in training_logs directory
    json_filename = f'{training_logs_dir}/training_history_{timestamp}.json'
    with open(json_filename, 'w') as f:
        json.dump(epoch_logs, f, indent=2)
    print(f"[OK] Epoch logs saved to {json_filename}", flush=True)
    
    # Also save a formatted text summary
    summary_filename = f'{training_logs_dir}/training_summary_{timestamp}.txt'
    with open(summary_filename, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {modelname} v{modelversion}\n")
        f.write(f"Feature Path: {featurepath}\n")
        f.write(f"Total Epochs Trained: {epoch_logs['epochs_trained']}/{training_epochs}\n")
        f.write(f"Best Val Loss: {epoch_logs['best_val_loss']:.6f} at Epoch {epoch_logs['best_val_loss_epoch']}\n")
        f.write(f"Final Train Loss: {epoch_logs['final_train_loss']:.6f}\n")
        f.write(f"Final Val Loss: {epoch_logs['final_val_loss']:.6f}\n")
        f.write(f"Final Train MAE: {epoch_logs['final_train_mae']:.6f}\n")
        f.write(f"Final Val MAE: {epoch_logs['final_val_mae']:.6f}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("EPOCH-BY-EPOCH METRICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Train MAE':<12} {'Val MAE':<12}\n")
        f.write("-" * 60 + "\n")
        for i in range(len(epoch_logs['loss'])):
            f.write(f"{i+1:<8} {epoch_logs['loss'][i]:<12.6f} {epoch_logs['val_loss'][i]:<12.6f} "
                   f"{epoch_logs['mae'][i]:<12.6f} {epoch_logs['val_mae'][i]:<12.6f}\n")
    print(f"[OK] Training summary saved to {summary_filename}", flush=True)
    
    print(f"   Total epochs trained: {epoch_logs['epochs_trained']}/{training_epochs}")
    print(f"   Best val_loss: {epoch_logs['best_val_loss']:.6f} at epoch {epoch_logs['best_val_loss_epoch']}")
    
    # Print training summary
    print("\n[TRAINING SUMMARY] Last 5 Epochs:")
    for i in range(max(0, len(epoch_logs['loss']) - 5), len(epoch_logs['loss'])):
        print(f"   Epoch {i+1:3d}: loss={epoch_logs['loss'][i]:.6f}, val_loss={epoch_logs['val_loss'][i]:.6f}, "
              f"mae={epoch_logs['mae'][i]:.6f}, val_mae={epoch_logs['val_mae'][i]:.6f}")
    
    ###################################################
    

    # Step 7: Evaluate model
    print("\n[Step 7] Evaluating model...")
    yhat = model.predict(X, verbose=0)
    
    # Calculate metrics
    mse = np.mean((y - yhat) ** 2)
    mae = np.mean(np.abs(y - yhat))
    rmse = np.sqrt(mse)
    
    # Calculate accuracy (percentage of predictions within 5% error)
    error_threshold = np.abs(y) * 0.05  # 5% of actual value
    accuracy = np.mean(np.abs(y - yhat) < error_threshold)
    
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"Accuracy (within 5% error): {accuracy*100:.2f}%")
    
    # Step 8: Save model and scaler info
    print("\n[Step 8] Saving model...")
    import tensorflow as tf
    import os
    import shutil
    import pickle
    
    # 1) Optional: keep .keras file (for local usage)
    model.save("./model.keras")
    print("[OK] Model saved as .keras format")
    
    # 2) Save normalization parameters for inference (needed to denormalize predictions)
    scaler_info = {
        'scaler_min': min_val,
        'scaler_max': max_val,
        'scaler_range': range_val
    }
    with open('./scaler_info.pkl', 'wb') as f:
        pickle.dump(scaler_info, f)
    print("[OK] Normalization parameters saved (for denormalization during inference)")
    


    # Exports the model as a TensorFlow SavedModel into a versioned folder
    # 3) REQUIRED for KServe: save as SavedModel in versioned directory structure
    # TensorFlow Serving expects: /mnt/models/1/saved_model.pb
    # Use model.export() instead of tf.saved_model.save() for proper TensorFlow Serving compatibility
    # Save to ./1/ so when we upload ./1, Model.zip will have 1/saved_model.pb
    os.makedirs("./1", exist_ok=True)
    # model.export() is the recommended method for TensorFlow Serving
    model.export("./1")
    print("[OK] Model saved successfully in SavedModel format at ./1/ using model.export()")
    




    # Upload metrics and model to Model Management Service
    # Step 9: Upload metrics and model to Model Management Service
    print("\n[Step 9] Uploading metrics and model to MMS...")
    
    artifactversion = "1.0.0"  # Start with 1.0.0 for suburban model
    
    # Prepare metrics (including training history)
    data = {
        'metrics': [
            {'MSE': str(mse)},
            {'MAE': str(mae)},
            {'RMSE': str(rmse)},
            {'Accuracy_5pct': str(accuracy)},
            {'Training_Samples': str(X.shape[0])},
            {'Feature': 'PRB_DL'},
            {'Resampling_Interval': '15 minutes'},
            {'Input_Steps': '192'},
            {'Output_Steps': '192'},
            {'Forecast_Horizon': '2 days'},
            {'Epochs_Trained': str(epoch_logs['epochs_trained'])},
            {'Early_Stopped': str(epoch_logs['early_stopped'])},
            {'Best_Val_Loss': str(min(epoch_logs['val_loss']))},
            {'Final_Train_Loss': str(epoch_logs['loss'][-1])},
            {'Final_Val_Loss': str(epoch_logs['val_loss'][-1])}
        ]
    }
    
    # Update artifact version
    url = f"http://modelmgmtservice.traininghost:8082/ai-ml-model-registration/v1/model-registrations/updateArtifact/{modelname}/{modelversion}/{artifactversion}"
    try:
        updated_model_info = requests.post(url).json()
        print(f"[OK] Artifact updated: {updated_model_info}")
    except Exception as e:
        print(f"Warning: Could not update artifact: {e}")
    
    # Upload metrics
    trainingjob_id = featurepath.split('_')[-1] if '_' in featurepath else featurepath
    mm_sdk.upload_metrics(data, trainingjob_id)
    print(f"[OK] Metrics uploaded for training job: {trainingjob_id}")
    
    # Copy training history to model directory BEFORE upload so it's included in Model.zip
    # This ensures training logs are archived with the model for KServe deployment
    import shutil
    import os
    files_copied = []
    host_training_logs_dir = '/home/ai1/DATA/LSTM/training_logs'
    
    try:
        # Copy JSON history
        shutil.copy(json_filename, './1/training_history.json')
        files_copied.append('training_history.json')
        
        # Copy summary text file
        shutil.copy(summary_filename, f'./1/training_summary_{timestamp}.txt')
        files_copied.append(f'training_summary_{timestamp}.txt')

        # Copy scaler so inference can use the exact training min/max
        if os.path.exists('./scaler_info.pkl'):
            shutil.copy('./scaler_info.pkl', './1/scaler_info.pkl')
            files_copied.append('scaler_info.pkl')
        
        print(f"[OK] Training history copied to model directory (will be included in Model.zip)")
        print(f"[OK] Files included: {', '.join(files_copied)}")
    except Exception as e:
        print(f"[WARNING] Could not copy training history to model directory: {e}")
    
    # Save epoch log file - if using volume mount, it's already on host at /workspace/training_logs
    # Otherwise, we can't copy to host from inside pod, so just note where it is
    print("\n[DEBUG] Checking epoch log file location...", flush=True)
    if os.path.exists(epoch_log_file):
        # If file was created in volume mount path, it's already on host
        if epoch_log_file.startswith('/workspace/training_logs'):
            print(f"[OK] Epoch log file saved to host via volume mount: {epoch_log_file}", flush=True)
        else:
            # File is in pod's local filesystem - note location (can't copy to host from here)
            print(f"[INFO] Epoch log file created at: {epoch_log_file}", flush=True)
            print(f"[INFO] File exists in pod but not accessible from host (no volume mount)", flush=True)
    else:
        print(f"[WARNING] Epoch log file not found at: {epoch_log_file}", flush=True)
    
#############################################




    # Upload model - upload the 1/ directory directly so Model.zip contains 1/saved_model.pb
    # This ensures when unpacked, it will be at /mnt/models/1/saved_model.pb
    # Model.zip will also include training_history.json and training_summary_{timestamp}.txt
    mm_sdk.upload_model("./1", modelname, modelversion, artifactversion)
    print(f"[OK] Model uploaded: {modelname}/{modelversion}/{artifactversion}")
    print(f"[OK] Model.zip includes: saved_model.pb, training_history.json, training_summary_{timestamp}.txt")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"Model expects: 192 input steps (2 days @ 15min intervals)")
    print(f"Model predicts: 192 output steps (2 days forecast)")
    print("=" * 60)


# step 2: The pod_spec_patch

#Defines the whole Kubeflow pipeline which is one step in the pipeline
# defining the pipeline function
@dsl.pipeline(
    name="Suburban_LSTM_Traffic_Prediction_Pipeline_GPU_V17",
    description="Multi-GPU LSTM (2 GPUs) optimized for speed: CPU threading pinned (inter_op=4, intra_op=8), Data caching, steps_per_execution=20, silent callbacks, Dropout (0.15), Batch=256/GPU, No Early Stopping (trains full epochs). Input: 192→192 steps. V17: Includes scaler_info.pkl in Model.zip for correct inference normalization",
)
def lstm_traffic_prediction_pipeline_suburban(
    featurepath: str, epochs: str, modelname: str, modelversion: str
):
    """
    GPU-enabled pipeline definition for suburban LSTM traffic prediction.
    
    Args:
        featurepath: Feature group name (e.g., "suburban_<trainingjob_id>")
        epochs: Number of training epochs
        modelname: Name of the model (e.g., "suburban-lstm-prbdl", S3-safe for MMS/MinIO)
        modelversion: Version of the model
    """
    
    # Create training task with GPU image
    # This huge JSON string contains Kubernetes pod settings for the training task.
    # It tells Kubernetes how to run the training container
    # Because training needs GPUs, CPU, memory, environment variables, and log storage.
    # GPU limits/requests: 2 GPUs, 16 CPUs, 32GB memory 
    # Environment variables for Configures CUDA/NVIDIA runtime behavior so TensorFlow can see and use the GPUs correctly
    # Mounts a host folder into the container so training logs are saved outside the container and remain accessible after the pod finishes
    
    trainop = train_export_model(
        featurepath=featurepath,
        epochs=epochs,
        modelname=modelname,
        modelversion=modelversion,
        pod_spec_patch='{"containers": [{"name": "main", "imagePullPolicy": "Always", "resources": {"limits": {"nvidia.com/gpu": "2", "cpu": "16", "memory": "32Gi"}, "requests": {"nvidia.com/gpu": "2", "cpu": "8", "memory": "16Gi"}}, "env": [{"name": "LD_LIBRARY_PATH", "value": "/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"}, {"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"}, {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "compute,utility"}], "volumeMounts": [{"name": "training-logs", "mountPath": "/workspace/training_logs"}]}], "volumes": [{"name": "training-logs", "hostPath": {"path": "/home/ai1/DATA/LSTM/training_logs", "type": "DirectoryOrCreate"}}], "runtimeClassName": "nvidia", "nodeSelector": {"nvidia.com/gpu.present": "true"}, "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]}'
    )
    # Turns off Kubeflow step caching. This is useful to avoid re-running the same step if the input has not changed.
    # Disable caching
    # GPU resources are injected via pod_spec_patch parameter above
    trainop.set_caching_options(False)
#####################################################


# step 3: compile the pipeline
# Compile GPU-enabled pipeline
# setting the name of the pipeline file
pipeline_func = lstm_traffic_prediction_pipeline_suburban
file_name = "lstm_traffic_prediction_pipeline_suburban_gpu_v17"

print(f"\nCompiling pipeline: {file_name}.yaml")
kfp.compiler.Compiler().compile(pipeline_func, f'{file_name}.yaml')
print(f"[OK] Pipeline compiled successfully: {file_name}.yaml")
#####################################################

# step 4: upload the pipeline to TM
# Upload pipeline to TM
print(f"\nUploading pipeline to Training Manager...")
pipeline_name = "Suburban_LSTM_Traffic_Prediction_Pipeline_GPU_V17"
pipeline_file = f'{file_name}.yaml'

###########################
# optional step but sometimes it's useful to delete old pipeline first to force fresh upload
# Try to delete old pipeline first to force fresh upload
try:
    print(f"Attempting to delete old pipeline version (if exists)...")
    delete_response = requests.delete(f"http://localhost:32002/pipelines/{pipeline_name}")
    if delete_response.status_code in [200, 204, 404]:
        print(f"[OK] Old pipeline deleted or didn't exist")
    else:
        print(f"[WARNING] Could not delete old pipeline (status: {delete_response.status_code})")
except Exception as e:
    print(f"[WARNING] Could not delete old pipeline: {e}")
    print("   Continuing with upload (will overwrite)...")
######################################
try:
    response = requests.post(
        f"http://localhost:32002/pipelines/{pipeline_name}/upload",
        files={'file': open(pipeline_file, 'rb')}
    )
    print(f"[OK] Pipeline uploaded successfully!")
    print(f"Response status: {response.status_code}")
    print(f"Response body: {response.text[:200]}")
    if response.status_code not in [200, 201, 204]:
        print(f"[WARNING] Upload returned status {response.status_code}, pipeline may not have updated!")
except Exception as e:
    print(f"Error uploading pipeline: {e}")
    print("You can upload it manually later using the TM API or GUI")

