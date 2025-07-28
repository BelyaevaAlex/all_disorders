# Neural Symbolic Time Series Classification (NSTSC) Project Documentation

## Project Overview
This project implements a Neural Symbolic Time Series Classification approach for EEG data analysis. The implementation is based on the NSTSC architecture and has been adapted for multi-channel EEG classification tasks.

## Data Structure
- Input data: EEG time series data from `/home/belyaeva.a/df_open_fixed.csv`
- Format: CSV file containing:
  - 19 EEG channels (F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fp1, Fp2, Fz, Cz, Pz)
  - Sampling frequency: 250 Hz
  - Labels indicating different conditions
- Data is automatically split into training (64%), validation (16%), and test (20%) sets

## Project Components

### Main Files
- `NSTSC_main.py`: Main entry point for training and evaluation
- `Models_node.py`: Contains neural network model definitions
- `utils/dataloader.py`: Data loading and preprocessing utilities
- `utils/train_utils.py`: Training helper functions

### Model Architecture
The project uses 6 different temporal logic neural networks:
1. TL_NN1: Conjunction of different predicates
2. TL_NN2: Disjunction of different predicates
3. TL_NN3: Always one predicate
4. TL_NN4: Eventually one predicate
5. TL_NN5: Always eventually one predicate
6. TL_NN6: Eventually always one predicate

Each EEG channel is processed independently through these networks, and the final prediction is made by combining predictions across all channels.

### Data Processing Pipeline
1. Load raw EEG data from CSV file
2. Convert string representations of EEG signals to numpy arrays
3. Split data into train/validation/test sets
4. Create sliding windows for temporal processing
5. Normalize each channel's data if specified
6. Train separate models for each channel
7. Combine predictions from all models and channels for final classification

### Training Process
- Each channel is processed independently
- For each channel, 6 different temporal logic networks are trained
- Models are trained using binary cross-entropy loss
- Best models are selected based on validation accuracy
- Final predictions are made by averaging predictions across all models and channels

## Usage
1. Ensure all dependencies are installed:
```bash
pip install pandas numpy scikit-learn torch
```

2. Run training:
```bash
python NSTSC_main.py
```

## Implementation Notes
- The code uses PyTorch for neural network implementation
- GPU acceleration is used if available
- Data normalization is performed per channel
- Sliding window approach is used for temporal feature extraction
- Model selection is based on validation performance

## Future Improvements
- Implement cross-validation
- Add support for different window sizes
- Add visualization of learned temporal patterns
- Optimize hyperparameters
- Add support for multi-class classification 