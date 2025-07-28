# Adapted from https://github.com/SquareResearchCenter-AI/BEExAI
import argparse
import glob
import os
import pandas as pd
import numpy as np
import seaborn as sns
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from train import Trainer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--save_path",
    type=str,
    default="output/benchmarks",
    help="Path to folder to save results",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="/home/belyaeva.a/features_open_new.csv",
    help="Path to the input CSV file",
)
args = parser.parse_args()

SAVE_PATH = args.save_path
if not os.path.exists(f"{SAVE_PATH}/models"):
    os.makedirs(f"{SAVE_PATH}/models")

# Force CPU usage
device = "cpu"
print("Device:", device)

# Load and preprocess data
print("Loading data from:", args.data_path)
data = pd.read_csv(args.data_path)

# Drop non-feature columns and get target
X = data.drop(columns=['patient_id', 'label', 'label_encoded'])
y = data['label_encoded'].astype(int)  # Ensure integer type for labels

DATA_NAME = os.path.splitext(os.path.basename(args.data_path))[0]
print("Dataset:", DATA_NAME)

with open(f"{SAVE_PATH}/models/{DATA_NAME}.txt", 'a') as f:
    for i, SEED in enumerate(range(10)):
        # Set random seeds
        torch.manual_seed(SEED)
        check_random_state(SEED)
        np.random.seed(SEED)
        
        # Split data into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=SEED, stratify=y
        )
        
        # Further split test into test/validation
        X_test, X_val, y_test, y_val = train_test_split(
            X_test, y_test, test_size=0.33, random_state=SEED, stratify=y_test
        )
        
        # Convert to numpy arrays
        X_train = X_train.values.astype(np.float32)
        X_test = X_test.values.astype(np.float32)
        X_val = X_val.values.astype(np.float32)
        
        num_labels = len(np.unique(y))
        n_samples, n_features = X_train.shape
        
        if i == 0:
            print(f"Labels: {num_labels}, Features: {n_features}, Train: {n_samples}, Test: {y_test.shape[0]}, Val: {y_val.shape[0]}", file=f)
            print(f"Labels: {num_labels}, Features: {n_features}, Train: {n_samples}, Test: {y_test.shape[0]}, Val: {y_val.shape[0]}")
        
        for MODEL_NAME in ["XGBClassifier", "BranchNet"]:
            # Set XGBoost parameters to use CPU
            if MODEL_NAME == "XGBClassifier":
                PARAMS = {"tree_method": "hist", "device": "cpu"}
            else:
                PARAMS = {}
                
            trainer = Trainer(MODEL_NAME, "classification", PARAMS, device)
            
            loaded = False
            if MODEL_NAME == "BranchNet":
                if glob.glob(f"{SAVE_PATH}/models/{DATA_NAME}_{MODEL_NAME}_{SEED}.pt"):
                    trainer.load_model(f"{SAVE_PATH}/models/{DATA_NAME}_{MODEL_NAME}_{SEED}.pt")
                    loaded = True
            else:
                if glob.glob(f"{SAVE_PATH}/models/{DATA_NAME}_{MODEL_NAME}_{SEED}.joblib"):
                    trainer.load_model(f"{SAVE_PATH}/models/{DATA_NAME}_{MODEL_NAME}_{SEED}.joblib")
                    loaded = True
            
            if not loaded:
                loss_file = f"{SAVE_PATH}/models/{DATA_NAME}_{MODEL_NAME}_{SEED}.png"
                trainer.train(X_train, y_train, X_val, y_val, SEED, loss_file=loss_file)
            
            if MODEL_NAME == "BranchNet":
                trainer.save_model(f"{SAVE_PATH}/models/{DATA_NAME}_{MODEL_NAME}_{SEED}.pt")
                trainer.model.eval()
            else:
                trainer.save_model(f"{SAVE_PATH}/models/{DATA_NAME}_{MODEL_NAME}_{SEED}.joblib")
            
            perf_metric = trainer.get_metrics(X_test, y_test)
            if MODEL_NAME == "BranchNet":
                print(MODEL_NAME, "performance", perf_metric, "hidden_neurons:", trainer.model.w1.data.shape[0], file=f)
                print(MODEL_NAME, "performance", perf_metric, "hidden_neurons:", trainer.model.w1.data.shape[0])
            else:
                print(MODEL_NAME, "performance", perf_metric, file=f)
                print(MODEL_NAME, "performance", perf_metric)
            torch.cuda.empty_cache()
    f.close()
