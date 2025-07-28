# NSTSC

This is the repository for the paper "[Neuro-symbolic Models for Interpretable Time Series Classification using Temporal Logic Description](https://arxiv.org/abs/2209.09114)", which is a neuro-symbolic model that leverages signal temporal logic and neural network to classify time series data and learn a human-readable, interpretable formula as explanation.

## Environment and Packages

> Python 3.9.7
> Numpy 1.20.3
> PyTorch 1.12.1
> Scikit-learn 0.24.2

## Dataset Description

This repository is implemented on a real-world wound healing data from mice experiments and a benchmark time series data repository -- UCR Time Series Classification Archive.

### Wound healing dataset
* Wound healing stages: hemostasis (Hem), inflammation (Inf), proliferation (Pro), maturation (Mat)
* Four protein features: MMP.2 ($𝑥^1$), IL.6 ($𝑥^2$), PLGF.2 ($𝑥^3$), VEGF ($𝑥^4$)
* 67 mice data, each with two time points for the first day and the date of healing stage
* SOTA models for comparison: FCN, ResNet, MLP for time series data

### UCR Time Series Archive
* The archive can be obtained as a zip file from [here](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)

## Model Training and Evaluation

> Please run [NSTSC_main.py](https://github.com/icdm22NSTSC/NSTSC/blob/main/Codes/NSTSC_main.py) for model training and evaluation.
> The Tree class in __NSTSC_main.py__ is the trained decision-tree based model.

