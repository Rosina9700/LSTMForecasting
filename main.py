#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from data_preparation import DataPreparation
from forecaster import LSTMModel
from forecast_eval import ForecastEvaluation


def plot_training_history(history):
    """
    Helper function to plot training and validation loss during LSTM training
    """
    plt.figure(figsize=(16,6))
    plt.plot(history['loss'],label='training_loss')
    plt.plot(history['val_loss'],label='validation_loss')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    pass

def plot_predictions(y_true, y_pred, intervals=None):
    """
    Helper function to plot true and prediction values at given t+n intervals
    """
    if intervals == None:
        intervals = [i*y_true.shape[1]/5 for i in range(0,5)]
    N = len(intervals)
    fig, axes = plt.subplots(N, figsize=(15, 6*N),sharey=True,sharex=True)
    for ax, interval in zip(axes, intervals):
        ax.set_title('t+{}'.format(interval+1))
        ax.plot(y_true[:,interval],label='true')
        ax.plot(y_pred[:,interval],label='pred')
        ax.legend(loc=1)
        ax.set_ylabel('kWh')
    plt.show()
    pass

def plot_evaluation(evaluation_loss, loss):
    '''
    Helper function to plot the evaluation values return for ForecastEvaluation
    '''
    plt.figure(figsize=(12,5))
    plt.plot(evaluation_loss)
    plt.ylabel(loss)
    plt.legend()
    plt.show()
    pass

def save_gridsearch_results(results, filename):
    columns=['n_units','sequence_length','dropout','final_layer_activation','min_val_loss','model']
    df = pd.DataFrame(results,columns=columns)
    df.to_csv(filename)
    return df

def plot_evalutions_loss(evaluation_loss, true, loss):
    fig, axes = plt.subplots(2, figsize=(15,12),sharex=True)
    axes[0].plot(true, label='true',color='b',alpha=0.75)
    axes[0].set_ylabel('energy demand kwh')
    axes[0].legend(loc='best')
    axes[1].plot(evaluation_loss[1], label='evaluation loss',color='r',alpha=0.75)
    axes[1].plot(evaluation_loss[0]*np.ones(len(true)),'--',label='average_loss',color='r',alpha=0.75)
    axes[1].set_ylabel(loss)
    axes[1].legend(loc='best')
    plt.show()
    pass

if __name__ =='__main__':
    print 'getting and preparing input data...'
    window = 48
    freq = '30T'
    dp = DataPreparation('~/git_hub/capstone_data/Azimuth/clean/project_6d8c_featurized.csv', 'energy_all',freq)
    df = dp.read_data()
    # resample_dict = defaultdict(list)
    # resample_dict['sum'] = ['energy_all','liq_precip']
    # resample_dict['mean'] = ['T','irr_glo']
    resample_dict = {'sum':['energy_all','liq_precip'], 'mean':['T','irr_glo']}
    df = dp.resample_columns(df, resample_dict)
    df = dp.create_time_features(df)
    df_X, df_y  = dp.prepare_data_multistep(df,window)
    X_train, X_test, y_train, y_test = dp.train_test_split(df_X.values, df_y.values, 0.1)

    print 'LSTM model training'
    units = np.arange(30,111,20)
    sequences = np.array([i * window for i in [1,3,5]])
    dropout = [0.0,0.2]
    activations = ['relu','tanh']
    # units=90
    # sequences=96
    # dropout=0.0
    # activations='tanh'
    batch_size = 25
    epochs = 80

    lm = LSTMModel(X_train, y_train, units, sequences, dropout, activations)
    lm.fit(batch_size,epochs,'checkpoint.keras')

    print 'Evaluating LSTM model'
    fe = ForecastEvaluation(lm)
    in_sample = fe.evaluate_in_sample(window*7)
    out_of_sample = fe.evaluate_out_of_sample(X_test, y_test, window*7)
    y_preds = lm.predict(X_test)
    plot_predictions(y_test,y_preds)
    plot_evaluation(in_sample[1],'mape')
    plot_evaluation(out_of_sample[1],'mape')

    df_res = save_gridsearch_results(lm.gridsearch_results, 'results_1206.csv')
