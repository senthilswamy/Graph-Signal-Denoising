# ==========================================================
# FILE: main.py
# ==========================================================
#
# Project:
# EVGAE-PKO Traffic Flow Denoising Framework
#
# Description:
# Main execution script for the proposed
# EVGAE-PKO traffic denoising model.
#
# Workflow:
#
# 1. Load dataset
# 2. Preprocess data
# 3. Construct graph representation
# 4. TRAIN_MODE check
#
#    TRAIN_MODE = 1
#       -> Run PKO optimization
#       -> Train EVGAE model
#       -> Save trained model
#       -> Save training history
#
#    TRAIN_MODE = 0
#       -> Load trained model
#       -> Generate denoised output
#       -> Save predictions
#
# Supported Datasets:
# - METR-LA
# - PEMS-BAY
#
# Output Files:
# - Trained Model (.h5)
# - Training History (.csv)
# - Denoised Signal (.csv)
#
# ==========================================================

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import os
import pandas as pd
import tensorflow as tf

from config import *

from data_loader import load_dataset

from objective_function import ObjectiveFunction

from model_builder import build_model

import optimization

# ==========================================================
# CREATE OUTPUT DIRECTORIES
# ==========================================================

os.makedirs("saved_models", exist_ok=True)

# ==========================================================
# DATASET SELECTION
# ==========================================================
#
# Available:
# "METR_LA"
# "PEMS_BAY"
#
# ==========================================================
DATASET_NAME = "PEMS_BAY"

# DATASET_NAME = "METR_LA"
TRAIN_MODE=0

print("\n======================================")
print("Selected Dataset :", DATASET_NAME)
print("======================================")

# ==========================================================
# LOAD DATASET
# ==========================================================

print("\nLoading Dataset...")

X, A_hat, num_nodes, numeric_cols = load_dataset(
    DATASETS[DATASET_NAME]
)

print("Dataset Loaded Successfully")

print("Input Shape :", X.shape)

print("Number of Nodes :", num_nodes)

# ==========================================================
# HYPERPARAMETER BOUNDS
# ==========================================================
#
# Parameters:
# [learning_rate,
#  batch_size,
#  gcn_units,
#  latent_dim,
#  dropout]
#
# ==========================================================

lb = [0.0001, 8, 16, 16, 0.1]

ub = [0.01, 64, 128, 128, 0.5]

dim = 5

# ==========================================================
# TRAINING MODE
# ==========================================================

if TRAIN_MODE == 1:

    print("\n======================================")
    print("TRAINING MODE ENABLED")
    print("======================================")

    # ------------------------------------------------------
    # Objective Function
    # ------------------------------------------------------

    objective = ObjectiveFunction(
        X,
        A_hat,
        num_nodes
    )
    # ==========================================================
    # PKO PARAMETERS
    # ==========================================================
    
    POP_SIZE = 30          # Number of candidate solutions
    MAX_ITER = 50          # Number of optimization iterations

    # ------------------------------------------------------
    # Run PKO Optimization
    # ------------------------------------------------------

    print("\nRunning PKO Optimization...")

    best_fitness, \
    best_position, \
    convergence_curve, \
     = optimization.PKO(

        POP_SIZE,

        MAX_ITER,

        lb,

        ub,

        dim,

        objective
    )

    print("\nOptimization Completed")

    # ------------------------------------------------------
    # Best Parameters
    # ------------------------------------------------------

    BEST_PARAMS = {

        "learning_rate":
            float(best_position[0]),

        "batch_size":
            int(best_position[1]),

        "gcn_units":
            int(best_position[2]),

        "latent_dim":
            int(best_position[3]),

        "dropout":
            float(best_position[4])
    }

    print("\nBest Parameters Found\n")

    for key, value in BEST_PARAMS.items():

        print(key, ":", value)

    # ------------------------------------------------------
    # Build Final Model
    # ------------------------------------------------------

    print("\nBuilding Final EVGAE Model...")

    model = build_model(

        best_position,

        A_hat,

        num_nodes
    )

    model.summary()

    # ------------------------------------------------------
    # Train Final Model
    # ------------------------------------------------------

    print("\nTraining Started...")

    history = model.fit(

        X,

        X,

        epochs=100,

        batch_size=BEST_PARAMS["batch_size"],

        validation_split=0.2,

        shuffle=True,

        verbose=1
    )

    print("\nTraining Completed")

    # ------------------------------------------------------
    # Save Model
    # ------------------------------------------------------

    model_path = (

        f"saved_models/"
        f"EVGAE_{DATASET_NAME}.h5"
    )

    model.save(model_path)
    print("\nModel Saved")

    print(model_path)

    # ------------------------------------------------------
    # Save Training History
    # ------------------------------------------------------

    history_df = pd.DataFrame(
        history.history
    )

    # history_path = (

    #     f"history/"
    #     f"{DATASET_NAME}_history.csv"
    # )

    history_df.to_csv(

        history_path,

        index=False
    )

    print("\nTraining History Saved")

    print(history_path)

    # ------------------------------------------------------
    # Save Optimization Results
    # ------------------------------------------------------

    optimization_results = pd.DataFrame({

        "Parameter": [

            "Learning Rate",

            "Batch Size",

            "GCN Units",

            "Latent Dimension",

            "Dropout",

            "Best Fitness",

            "Execution Time"
        ],

        "Value": [

            BEST_PARAMS["learning_rate"],

            BEST_PARAMS["batch_size"],

            BEST_PARAMS["gcn_units"],

            BEST_PARAMS["latent_dim"],

            BEST_PARAMS["dropout"],

            best_fitness,

            execution_time
        ]
    })

    # optimization_results.to_csv(

    #     f"history/"
    #     f"{DATASET_NAME}_best_parameters.csv",

    #     index=False
    # )

    print("\nOptimization Results Saved")

# ==========================================================
# TESTING MODE
# ==========================================================

else:

    print("\n======================================")
    print("TEST MODE ENABLED")
    print("======================================")

    # ------------------------------------------------------
    # Load Saved Model
    # ------------------------------------------------------

    model = build_model(
        [0.001,32,64,64,0.3],# from pko optimization
        A_hat,
        num_nodes
        )
    
    model.load_weights(f"saved_models/EVGAE_{DATASET_NAME}_weights.h5")


    print("Model Loaded Successfully")

    # ------------------------------------------------------
    # Generate Denoised Output
    # ------------------------------------------------------

    print("\nGenerating Denoised Signal...")

    denoised_signal = model.predict(X)

    denoised_signal = denoised_signal.squeeze()

    print("Prediction Completed")
    
  