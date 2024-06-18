# Air Quality Analysis Using Neural Networks

This Jupyter notebook explores air quality analysis by predicting carbon monoxide (CO(GT)) levels using various neural network architectures, including Multi-layer Perceptrons (MLP). The experiments focus on both model selection and the evaluation of different training algorithms.

## Dataset and Knowledge Preparation

- **Data Source**: The dataset includes 3304 instances of hourly averaged measurements from a multisensor device placed at road level in a polluted city. The sensors include spectrometer analyzers (GT), solid state metal oxide detectors (PT08.Sx), temperature, relative humidity, and absolute humidity sensors.
- **Objective**: Predict the continuous value of CO(GT) which represents the carbon monoxide levels.
- **Missing Data**: Features with missing data are flagged with `-999` and require preprocessing for accurate analysis.

## Prerequisites

Ensure the following libraries are installed:
- NumPy
- Pandas
- Matplotlib
- scikit-learn

```bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
```

## Usage

### Setup
- Ensure all dependencies are installed.
- Download the dataset (`sensor_data.xlsx`) and the Jupyter notebook into the same directory.

### Execution
- Open the notebook in a Jupyter environment.
- Run the notebook cells sequentially to perform data preprocessing, model training, evaluation, and comparison of different neural network architectures.

## Experiments

### Simple MLP Model Selection
- **Models Tested**: Various MLP configurations with one or two hidden layers and different activation functions (`relu`, `logistic`).
- **Methodology**: Utilize `GridSearchCV` for hyperparameter tuning and model selection based on Mean Squared Error (MSE).

### Training Algorithm Comparison
- **Algorithms Compared**: Stochastic Gradient Descent (SGD) and ADAM optimization.
- **Analysis**: Evaluate the performance of both training algorithms over 300 iterations, comparing training loss and MSE on both training and testing datasets.

## Results
- **Model Parameters**: Output the best model parameters and the cross-validation MSE with its standard deviation.
- **Model Evaluation**: Re-train the selected model using the whole training set and evaluate its MSE and R² score on the testing set.
- **Training Dynamics**: Discuss the differences in training dynamics and final performance between SGD and ADAM as optimization algorithms.

## Model Development
- **Robust MLP Regressor**: Develop an MLP regressor that accounts for missing and noisy features.
- **Considerations**: Include handling missing features, noisy data, and a robust model selection practice to ensure generalization.

## Contributing
- Contributions to improve the analysis or extend the functionality are welcome. Please fork the repository, make your changes, and submit a pull request.

## Conclusion
- This project demonstrates the use of neural networks in environmental science to predict air quality based on complex sensor data. The analysis provides insights into effective model architectures and training algorithms for real-world data with challenges like missing values and noise.
