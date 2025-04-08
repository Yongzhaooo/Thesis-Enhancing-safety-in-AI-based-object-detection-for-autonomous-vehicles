# Thesis Project: Enhancing safety in AI-based object detection for autonomous vehicles through out-of-distribution monitoring

This repository contains scripts and tools that we used for the master thesis: Enhancing safety in AI-based object detection for autonomous vehicles through out-of-distribution monitoring

Pub:
https://odr.chalmers.se/items/56a3c3f4-1150-49e1-9782-af4e090b16de
---

## Configuration

1. Navigate to the `/config` directory and edit the `config.yaml` file.
2. Update the paths in the configuration file to match your directory structure.

---

## Main Workflow

1. The main script will:
   - Use the `top_level_layer` parameter to generate subfolders containing noisy images.
   - Perform evaluation on the noisy images.
   - Delete the noisy images after evaluation.

2. **Note**: 
   - If you want to **keep the noisy images**, you need to modify the corresponding class in the script.

---

## Visualization

After completing the evaluation, use the following scripts under the `/drawtools` directory to generate plots:

1. **`EvaluationPlotter.py`**:
   - Generates evaluation plots for Gaussian and Mosaic noise effects.

2. **`HistogramPlotter.py`**:
   - Creates histograms to visualize bias distributions across datasets.

---

## Hardware Information

To ensure reproducibility, the following hardware and driver information was used during the development and testing of this project:

### System Information
- **Operating System**: Ubuntu 22.04.4 LTS
- **CPU**: Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz
- **Logical CPUs**: 20
- **Physical CPUs**: 10
- **CPU Frequency**: 5300.00 MHz
- **Memory**: 33.49 GB (Total), 20.68 GB (Available)

### GPU Information
- **GPU Model**: NVIDIA Quadro RTX 4000
- **Driver Version**: 535.183.01
- **Memory**: 8192 MB (Total), 911 MB (Used), 7053 MB (Free)

---

## Setting Up the Environment

To recreate the Conda environment used in this project, follow these steps:

1. Ensure you have Conda installed on your system.
2. Clone this repository:



