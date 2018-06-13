# Forecasting with Keras LSTM RNN
In this study, I investigate the use of LSTM RNNs to forecast a day ahead of 15minute energy demand (96 time steps).
External features include weather variable of temperature, irradiance, rainfall and humidity.

There are two scenarios tested in this study:
- No forecast available for external weather variables and the LSTM RNN architecture is designed for multi-step (96 steps) prediction
- Forecast data for weather variables is available and the LSTM RNN architecture is designed for single-step forecast at each 15 interval. Therefore features for a given demand forecast is the forecast data plus the energy demand 24 hours ago. 

The performance of the two approaches is evaluated using mean MAPE over a full day

## Tools
The keras deep learning library is used
