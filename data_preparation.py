import pandas as pd
import numpy as np

class DataPreparation(object):
    '''
    Prepares the choosen data for the LSTMModel forecaster.
    Data must already exist in .csv format.
    '''
    def __init__(self, path, target_name, freq):
        '''
        INPUTS:
        --------------------------
        path: String, locating data in a csv format
        target_name: String, column name for target variable.
        '''
        self.path = path
        self.target_name = target_name
        self.freq = freq

    def read_data(self):
        '''
        Read data from specified location and handle missing values
        RETURNS:
        --------------------------
        self
        '''
        df = pd.read_csv(self.path)
        df = df[['t','energy_all','T','liq_precip','irr_glo']]
        df['t'] = pd.to_datetime(df['t'])
        df['energy_all'] = df['energy_all']/1000. #to get energy in kWh rather than Wh
        df.set_index('t', inplace=True, verify_integrity=True)
        return df

    def resample_columns(self, df, resample_dict):
        for key, columns in resample_dict.iteritems():
            if key == 'sum':
                df_sum = df[columns].resample(self.freq).sum()
            else:
                df_mean = df[columns].resample(self.freq).mean()
        df = df_sum.join(df_mean, how='inner')
        return df



    def fill_nans(self, df):
        '''
        Hande NaNs in provided data.
        Target:
        Leading and trailing NaNs from target are removed.
        Small missing periods filled with linear interpolation
        Features:
        Assume that we can fill gaps with linear interpolation
        Trailing and leading NaNs handled with back and forward fill
        INPUTS:
        --------------------------
        df: Pandas DataFrame
        RETURNS:
        --------------------------
        df: Pandas DataFrame
        '''
        # remove leading and trailing NaNs
        first_idx = df[self.target_name].first_valid_index()
        last_idx = df[self.target_name].last_valid_index()
        df = df.loc[first_idx:last_idx,:]
        # small periods of missing data for actual_kwh (<5 timesteps) Linear interpolate
        df[self.target_name].interpolate('linear',inplace=True)
        # lots of missing T data. Linear interpolate this too
        df.interpolate('linear',inplace=True)
        # handling missing data at beginning and end of series
        df.fillna(df.bfill(),inplace=True)
        df.fillna(df.ffill(),inplace=True)
        return df

    @staticmethod
    def create_time_features(df):
        '''
        Create additional features and reduce dataframe to target and features only
        RETURNS:
        --------------------------
        df: Pandas DataFrame
        '''
        # Modelling hour as a continuous variable so need to handle to discontinuity between 23 and 0.
        # Splitting into two feature of x and y length if projected onto a 2d-axis

        df['x_y_hour'] = np.sin(2*np.pi*df.index.hour/24.) + np.cos(2*np.pi*df.index.hour/24.)
        df['day_of_year'] = df.index.dayofyear
        df['dow'] = df.index.dayofweek
        df = pd.get_dummies(df, columns=['dow'])
        return df

    def prepare_data_multistep(self, df, N):
        """
        Prepare data for a model that forecasts N timesteps in the future
        INPUTS:
        --------------------------
        df: Pandas DataFrame
        N: Integer
        RETURNS:
        --------------------------
        df: Pandas DataFrame
        df_target: Pandas DataFrame
        """
        target_cols = []
        for i in xrange(1,N+1):
            name = self.target_name +'+'+str(i)
            df[name] = df[self.target_name].shift(-i)
            target_cols.append(name)
        df.dropna(inplace=True)
        # create X and y arrays
        df_target = df[target_cols]
        df.drop(target_cols, axis=1, inplace=True)
        return df, df_target

    @staticmethod
    def train_test_split(X, y, percentage):
        """
        Split data into training and test set given split percentage
        INPUTS:
        --------------------------
        X, y: numpy ndarray
        RETURNS:
        --------------------------
        X_train, X_test, y_train, y_test: numpy ndarrays
        """
        n_test = int(percentage * X.shape[0])
        X_train, y_train = X[:-n_test], y[:-n_test]
        X_test, y_test = X[-n_test:], y[-n_test:]
        return X_train, X_test, y_train, y_test
