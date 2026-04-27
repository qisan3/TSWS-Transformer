# Two-Stage Weakly Supervised Transformer for Machine Tool Wear Prediction in PHMAP 2025 Data Challenge


This repository contains the official codebase and submission environment for our approach to the **[PHMAP 2025 Data Challenge]**. 

## 🗂️ Repository Structure

To facilitate both academic review and challenge deployment, this repository is organized into two primary environments:

```text
.
├── tcdata/                   # Raw challenge data (Controller_Data and Sensor_Data)
├── processed_data/           # Output directory for Sktime-formatted .ts files
├── experiments/              # Logs, checkpoints, and metrics from training
├── feature_extraction.py     # End-to-end data processing and windowing script
│
├── training_src/             # 🧠 MODEL TRAINING ENVIRONMENT
│   ├── datasets/             # Data loading and imputation logic
│   ├── models/               # Transformer architectures and custom loss functions
│   ├── main.py               # Main training entry point
│   └── running.py            # Epoch execution and validation logic
│
└── submission_docker/        # 🐳 DEPLOYMENT & INFERENCE ENVIRONMENT
    ├── Dockerfile            # Docker configuration for challenge submission
    ├── run.sh                # Execution script required by the platform
    ├── main.py               # Inference entry point
    ├── lib/                  # Inference dependencies (feature extraction, data loader)
    └── model/                # Pre-trained weights and normalization parameters
