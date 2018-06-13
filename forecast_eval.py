import numpy as np

class ForecastEvaluation(object):
    '''
    Class to evaluate the given forecaster class.
    '''
    def __init__(self, forecaster, evaluation_metric='mean_absolute_percentage_error'):
        '''
        INPUTS:
        --------------------------
        forecaster: LSTMModel instance
        evaluation_metric: String
        '''
        self.forecaster = forecaster
        self.valid_metrics = ['mean_absolute_percentage_error','mean_squared_error']
        if evaluation_metric in self.valid_metrics:
            self.evaluation_metric = evaluation_metric
        else:
            print 'Invalid evaluation metric'

    @staticmethod
    def mape(y_true, y_pred):
        '''
        Return the mean absolute percentage error across axis 1 between
        2 2-D numpy arrays.
        INPUTS:
        --------------------------
        y_true, y_pred: numpy 2D-arrays
        RETURNS:
        --------------------------
        numpy 1D-array
        '''
        return ((np.abs(y_true - y_pred)/y_true).mean(axis=1))*100

    @staticmethod
    def mse(y_true, y_pred):
        '''
        Return the mean sqaured error across axis 1 between
        2 2-D numpy arrays.
        INPUTS:
        --------------------------
        y_true, y_pred: numpy 2D-arrays
        RETURNS:
        --------------------------
        numpy 1D-array
        '''
        return (np.power(y_true - y_pred,2)).mean(axis=1)

    def evaluate(self, y_true, y_pred):
        '''
        MAPE or MSE for each forecast period. Returns both the average MAPE or
        MSE across the whole period as well as the MAPE or MSE for each
        multistep forecast period.
        INPUTS:
        --------------------------
        y_true, y_pred: numpy 2D-arrays
        RETURNS:
        --------------------------
        float, numpy 1D-array
        '''
        if self.evaluation_metric == 'mean_absolute_percentage_error':
            results = self.mape(y_true, y_pred)
        elif self.evaluation_metric == 'mean_squared_error':
            results = self.mse(y_true, y_pred)

        if len(results) > 1:
            # if forecasting for more than one timestep, find the mean of means.
            return results.mean(), results
        elif len(results) == 1:
            # if forecasting for one timestep, take the single mean.
            return results[0], results

    def evaluate_in_sample(self, window=None):
        '''
        Evaluates the predictions for a period of data that was included
        in the training of the forecaster.
        Window corresponds to the period of data from the end of the dataset
        to evaluate.
        INPUTS:
        --------------------------
        window: Integer
        RETURNS:
        --------------------------
        float, numpy 1D-array
        '''
        if window==None:
            window == self.forecaster.custom_mse_sequence_length
        X_train, X_val, y_train, y_val = self.forecaster.create_validation_split()
        y_pred = self.forecaster.predict(X_train)
        return self.evaluate(y_train[-window:], y_pred[-window:])

    def evaluate_out_of_sample(self, X, y, window=None):
        '''
        Evaluates the forecasts for a period of data that was not included in
        the training of the forecaster and provided by the user.
        window corresponds to the period of data from the end of the dataset
        to evaluate.
        INPUTS:
        --------------------------
        window: Integer
        RETURNS:
        --------------------------
        float, numpy 1D-array
        '''
        if window==None:
            window == self.forecaster.custom_mse_sequence_length
        y_pred = self.forecaster.predict(X)
        return self.evaluate(y[-window:], y_pred[-window:])
