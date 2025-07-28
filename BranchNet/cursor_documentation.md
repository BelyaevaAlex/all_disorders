# BranchNet Project Documentation

## Project Overview
BranchNet is a neuro-symbolic learning framework that converts decision tree ensembles into sparse, partially connected feedforward neural networks. Each branch in the tree ensemble corresponds to a hidden neuron, preserving symbolic structure while enabling gradient-based learning.

## Project Structure
- `BranchNet.py`: Core BranchNet neural network architecture definition
- `BranchNetFramwork.py`: BranchNet learning framework implementation
- `train.py`: Main script for building and training models
- `benchmetrics.py`: Training configurations and benchmark experiments
- `requirements.txt`: Python dependencies
- `output/`: Directory for saving experiment results

## Environment Setup
The project requires Python 3.10 or higher and can be run in a virtual environment:
- Current working environment: `/home/belyaeva.a/venv_new`
- Activation command: `source /home/belyaeva.a/venv_new/bin/activate`

## Data
- Original data path: Uses OpenML datasets (downloaded via openml_download.py)
- Current custom dataset: `/home/belyaeva.a/features_open_new.csv`

## Running Experiments
1. Activate the virtual environment:
   ```bash
   source /home/belyaeva.a/venv_new/bin/activate
   ```
2. Run benchmarks:
   ```bash
   python benchmetrics.py
   ```

## Notes and TODOs
1. The benchmetrics.py script currently uses hardcoded paths and OpenML datasets. It needs modification to work with custom CSV files.
2. The script requires proper data preprocessing to match the expected format from OpenML datasets.

## Recent Changes
- Added support for custom CSV dataset at `/home/belyaeva.a/features_open_new.csv` 