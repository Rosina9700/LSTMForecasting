import numpy as np
import tensorflow as tf

from keras import Sequential
from keras.layers import GRU, Dense, Dropout, LSTM
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler


class LSTMModel(object):
    '''
    LSTM Forecaster class.
    Handles data preparation (splitting and scaling),
    building and training LSTM models,
    gridsearch over given parameter space,
    prediction and evaluation.
    '''
    def __init__(self, X, y, n_units=10, sequence_length=24, dropout=0.0,
                final_layer_activation='relu'):
        '''
        INPUTS:
        --------------------------
        X, y: numpy ndarrays
        n_units: Integer or List-like
        sequence_length: Integer or List-like
        dropout: Float or List-like
        final_layer_activation: String or List-like
        '''
        self.parameters = {}
        self.parameters['n_units'] = n_units
        self.parameters['sequence_length'] = sequence_length
        self.parameters['dropout'] = dropout
        self.parameters['final_layer_activation'] = final_layer_activation
        self.custom_mse_sequence_length = None
        self.best_param_combination = None
        self.gridsearch_results = None
        self.path_checkpoint = None
        self.model = None
        self.history = None
        self.data = {'X_scaler': MinMaxScaler(),
                     'y_scaler': MinMaxScaler(),
                     'X': X,
                     'y': y,
                     'validation_split': 0.1}

    def create_validation_split(self):
        '''
        Create validation split given validation split percentage.
        RETURNS:
        --------------------------
        X_train, X_test, y_train, y_train: numpy ndarrays
        '''
        n_test = int(self.data['validation_split'] * self.data['X'].shape[0])
        X_train, y_train = self.data['X'][:-n_test], self.data['y'][:-n_test]
        X_test, y_test = self.data['X'][-n_test:], self.data['y'][-n_test:]
        return X_train, X_test, y_train, y_test

    def scale_data(self, X_train, X_val, y_train, y_val):
        '''
        Scaling data using MinMaxScaler.
        INPUTS:
        ---------------------------
        X_train, X_val, y_train, y_val: numpy ndarrays
        RETURNS:
        ---------------------------
        X_train, X_val, y_train, y_val: numpy ndarrays
        '''
        # fit scalers and transform
        X_train = self.data['X_scaler'].fit_transform(X_train)
        y_train = self.data['y_scaler'].fit_transform(y_train)
        X_val = self.data['X_scaler'].transform(X_val)
        y_val = self.data['y_scaler'].transform(y_val)
        return X_train, X_val, y_train, y_val

    def prepare_training_data(self):
        '''
        Prepare data for LSTM forecaster and reshapes validation set
        RETURNS:
        --------------------------
        X_train, y_train: numpy ndarrays
        validation_data: Tuple of numpy ndarrays
        '''
        X_train, X_val, y_train, y_val = self.create_validation_split()
        X_train, X_val, y_train, y_val = self.scale_data(X_train, X_val, y_train, y_val)

        # reshaping validation set
        validation_data = [np.expand_dims(X_val, axis=0),
                           np.expand_dims(y_val, axis=0)]
        return X_train, y_train, validation_data

    @staticmethod
    def batch_generator(X, y, batch_size, sequence_length):
        '''
        Create batch generator for random sequences of data for training the LSTM
        YIELDS
        --------------------------
        Tuple of numpy ndarrays
        '''
        while True:
            # initiate input data arrays
            x_shape = (batch_size, sequence_length, X.shape[1])
            y_shape = (batch_size, sequence_length, y.shape[1])
            x_batch = np.zeros(shape=x_shape, dtype=np.float16)
            y_batch = np.zeros(shape=y_shape, dtype=np.float16)

            for i in xrange(batch_size):
                # fill arrays
                idx = np.random.randint(X.shape[0] - sequence_length)
                x_batch[i] = X[idx:idx+sequence_length]
                y_batch[i] = y[idx:idx+sequence_length]

            yield (x_batch, y_batch)

    def custom_mse(self, y_true, y_pred):
        '''
        Mean square error with N timesteps ignores from the beginning of the sequence.
        LSTMs typically don't perform well on timeseries with length << training sequence length
        Therefore ignore these and focus on improving fit at later timesteps
        RETURNS
        --------------------------
        loss_mean: tensorflow reduced_mean matrix
        '''
        N = self.custom_mse_sequence_length/4
        y_true_slice = y_true[:, N:, :]
        y_pred_slice = y_pred[:, N:, :]
        loss = tf.losses.mean_squared_error(labels=y_true_slice,
                                            predictions=y_pred_slice)
        loss_mean = tf.reduce_mean(loss)
        return loss_mean

    def build_model(self, n_units, dropout, final_layer_activation):
        '''
        Build single layer LSTM architecture according the Class attributes.
        INPUTS:
        --------------------------
        n_units: number of LSTM units, Integer
        dropout: dropout fraction, Float
        final_layer_activiation: activation for final Dense layer, String
        RETURNS:
        --------------------------
        self
        '''
        n_inputs = self.data['X'].shape[1]
        n_outputs = self.data['y'].shape[1]
        model = Sequential()
        model.add(LSTM(units=n_units, return_sequences=True, input_shape=(None,n_inputs,)))
        model.add(Dropout(dropout))
        model.add(Dense(n_outputs, activation=final_layer_activation))
        optimizer = RMSprop(lr=1e-2)
        model.compile(loss=self.custom_mse, optimizer=optimizer)
        self.model = model
        return self

    @staticmethod
    def create_callbacks(path_checkpoint):
        '''
        Create callbacks to mitigate chances of overfitting and
        converging to a local minimum and to reduce training time.
        INPUTS:
        --------------------------
        path_checkpoint: filename to save the checkpoint to, String
        RETURNS:
        --------------------------
        callbacks: List of Callbacks
        '''
        # create checkpoint each time validation loss improves
        callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                              monitor='val_loss',
                                              verbose=1,
                                              save_weights_only=True,
                                              save_best_only=True)
        # Stop early if validation loss doesn't improve
        callback_early_stopping = EarlyStopping(monitor='val_loss',
                                                patience=5,
                                                verbose=1)
        # reduce learning rate if validation loss doesn't improve
        callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                                factor=0.1,
                                                min_lr=1e-5,
                                                patience=3,
                                                verbose=1)
        callbacks = [callback_checkpoint, callback_early_stopping, callback_reduce_lr]
        return callbacks

    def single_param_check(self):
        '''
        Check if model architecture hyperparameters are single values or
        lists for gridsearch
        RETURNS:
        --------------------------
        Boolean
        '''
        for key, param in self.parameters.iteritems():
            if isinstance(param, list)|isinstance(param, np.ndarray):
                if len(param) > 1:
                    return False
                else:
                    self.parameters[key] = param[0]
        return True

    def train_model(self, X_train, y_train, validation_data, batch_size, epochs, n_units, sequence_length, dropout, final_layer_activation):
        '''
        For the given data and model architecture,
        this function builds the model and trains.
        INPUTS:
        --------------------------
        X_train, y_train: numpy ndarrays
        validation_data: Tuple of numpy ndarrays
        batch_size: Integer
        epochs: Integer
        n_units: Integer
        sequence_length: Integer
        dropout: Float
        final_layer_activation: String
        RETURNS:
        --------------------------
        history: Keras callback History object
        '''
        print 'building RNN...'
        self.custom_mse_sequence_length = sequence_length
        self.build_model(n_units,dropout,final_layer_activation)
        steps_per_epoch = (X_train.shape[0]-sequence_length)/(sequence_length)
        callbacks = self.create_callbacks(self.path_checkpoint)
        generator = self.batch_generator(X_train, y_train, batch_size, sequence_length)
        print 'training RNN...'
        history = self.model.fit_generator(generator=generator,
                           epochs=epochs,
                           steps_per_epoch=steps_per_epoch,
                           validation_data=validation_data,
                           callbacks=callbacks,
                           use_multiprocessing=True)
        return history

    def gridsearch(self,  X_train, y_train, validation_data, batch_size, epochs):
        '''
        Calls train_model for every combination of parameters.
        Saves set up, minimum loss on validation set and model.
        INPUTS:
        --------------------------
        X_train, y_train: numpy ndarrays
        validation_data: Tuple of numpy ndarrays
        batch_size: Integer
        epochs: Integer
        RETURNS:
        --------------------------
        self
        '''
        combinations = np.asarray(np.meshgrid(self.parameters['n_units'],
                                              self.parameters['sequence_length'],
                                              self.parameters['dropout'],
                                              self.parameters['final_layer_activation'])).T.reshape(-1,4)
        print '{} combinations to check'.format(combinations.shape[0])
        search_results = []
        counter = 1
        for p in combinations:
            print '*************************'
            print 'permutation {}/{}'.format(counter,combinations.shape[0])
            print 'setup: n_units={}, sequence_length={}, dropout={}, activation={}'.format(p[0],p[1],p[2],p[3])
            history = self.train_model(X_train, y_train, validation_data,
                                       batch_size, epochs,
                                       int(p[0]), int(p[1]),
                                       float(p[2]), p[3])
            min_val_loss = min(history.history['val_loss'])
            search_results.append([int(p[0]), int(p[1]), float(p[2]), p[3], min_val_loss, self.model])
            counter += 1
        self.gridsearch_results = search_results
        return self

    def fit(self, batch_size, epochs, path_checkpoint, validation_split=0.1):
        '''
        Fit model function performs the following:
        - prepares data
        - if single value parameters are given, builds and trains model
        - if multiple value parameters are given, gridsearch and choose best model
        INPUTS:
        --------------------------
        batch_size: Integer
        epochs: Integer
        path_checkpoint: String
        validation_split: Float
        RETURNS:
        --------------------------
        self
        '''
        # prepare data by creating validation split, scaling and reshaping
        self.data['validation_split'] = validation_split
        self.path_checkpoint = path_checkpoint
        X_train, y_train, validation_data = self.prepare_training_data()

        if self.single_param_check():
            self.custom_mse_sequence_length = self.parameters['sequence_length']
            self.history = self.train_model(X_train, y_train, validation_data,
                                            batch_size, epochs,
                                            self.parameters['n_units'],
                                            self.parameters['sequence_length'],
                                            self.parameters['dropout'],
                                            self.parameters['final_layer_activation'])
        else:
            self.gridsearch(X_train, y_train, validation_data, batch_size, epochs)
            best = min(self.gridsearch_results, key=lambda x: x[4])
            self.model = best[5]
            self.best_param_combination = {'n_units':best[0],
                                           'sequence_length': best[1],
                                           'dropout': best[2],
                                           'final_layer_activation': best[3]}
            self.history = self.train_model(X_train, y_train, validation_data,
                                            batch_size, epochs,
                                            best[0],
                                            best[1],
                                            best[2],
                                            best[3])
        try:
            self.load_weights(path_checkpoint)
        except Exception as error:
            print 'Error trying to load checkpoint'
            print error

        return self

    def load_weights(self, path):
        '''
        loads weights from a given checkpoint
        INPUTS:
        --------------------------
        path: String
        RETURNS:
        --------------------------
        self
        '''
        self.model.load_weights(path)
        return self

    def evaluate(self, X, y):
        '''
        Evaluates the performance of the model in test mode on a given dataset.
        Evaluation metric is the loss function from model training.
        INPUTS:
        --------------------------
        X, y: numpy ndarrays
        RETURNS:
        --------------------------
        float
        '''
        X_scaled = self.data['X_scaler'].transform(X)
        y_scaled = self.data['y_scaler'].transform(y)
        data = [np.expand_dims(X_scaled, axis=0),
                np.expand_dims(y_scaled, axis=0)]

        return self.model.evaluate(data[0], data[1])

    def predict(self, X):
        '''
        Computes predictions from the model in test mode for the given data.
        INPUTS:
        --------------------------
        X: numpy ndarray
        RETURNS:
        --------------------------
        numpy ndarray
        '''
        X_scaled = self.data['X_scaler'].transform(X)
        data = np.expand_dims(X_scaled, axis=0)
        predictions = self.model.predict(data)[0]
        y_preds = self.data['y_scaler'].inverse_transform(predictions)
        return y_preds

    def forecast(self, X):
        '''
        Takes the final prediction from a timeseries input to
        provide a single forecast.
        INPUTS:
        --------------------------
        X: numpy ndarray
        RETURNS:
        --------------------------
        numpy ndarray
        '''
        return self.predict(X)[-1,:]
