# Forecasting with Keras LSTM RNN
In this study, I investigate the use of LSTM RNNs to forecast a day ahead of 30minute energy demand (48 time steps).
External features include weather variable of temperature, irradiance and rainfall.

Energy demand is from a C&I facility based in Accra, Ghana.
Hourly weather data has been taken from the NASA MERRA2 dataset and downsampled from 1hour intervals to 30minute intervals.

It is assumed that no forecast available for external weather variables and the LSTM RNN architecture is designed for multi-step (96 steps) prediction.

The performance is evaluated using mean MAPE over a full day

## Dependancies
- keras
- numpy
- pandas
- matplotlib

## Scripts
**data_preparation.py**
DataPreparation Class to prepares the data for the LSTMModel forecaster.  This includes:
- Resampling
- Handling missing values
- Creating temporal features
- Create multistep target
- Train / Test split

**forecaster.py**
LSTMModel forecaster class which performs a gridsearch and/or trains model.
- Grid search can be performed across a limited feature space at present:
  - Number of LSTM unit
  - Sequence length for training batches
  - Dropout fraction
  - Final layer activation
- A custom mse loss function is used where a window at the beginning of the predictions is ignored during evaluation. This is due to the fact that LSTM learn from the sequence and therefore perform poorly at the beginning. The window is proportional to the sequence length.
- A single layer LSTM is used at present for simplicity but easy to expand.
- RMSProp optimizer wiht ReduceLROnPlateau callbacks implemented for optimization as this is generally accepted as a powerful optimizer for LSTMs.
- ModelCheckpoint callback implemented to be able to reload model weights which correspond to lowest validation loss.
- EarlyStopping callback implemented to avoid overfitting and reduce training time

**forecast_eval.py**
ForecastEvaluation class to evaluate the performance of the LSTMModel forecaster.
- Allows evaluation using mean square error or mean square error
- Performs in and out of sample evaluations
