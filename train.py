import keras
from keras.engine.sequential import relax_input_shape
from keras.layers.core import Dropout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from utils import *

ASSET_NAME =  "ALTHUOB/BTC-CXDX"
# ASSET_NAME =  "Z-CRY/IDX"

STEP_SIZE = 1000

# FOR REPRODUCIBILITY
np.random.seed(7)

# IMPORTING DATASET
dataset = read_data_by_asset_name(asset_name=ASSET_NAME)

# CREATING OWN INDEX FOR FLEXIBILITY
obs = np.arange(1, len(dataset) + 1, 1)

# TAKING DIFFERENT INDICATORS FOR PREDICTION
training_data = dataset[['Close']]


# PREPARATION OF TIME SERIES DATASE
training_data = np.reshape(training_data.values, (len(training_data),1)) # 1664
scaler = MinMaxScaler(feature_range=(0, 1))
training_data = scaler.fit_transform(training_data)


# TIME-SERIES DATASET (FOR TIME T, VALUES FOR TIME T+1)
trainX, trainY = new_dataset(training_data, STEP_SIZE)

# RESHAPING TRAIN AND TEST DATA
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

# LSTM MODEL
model = Sequential()
model.add(LSTM(128, input_shape=(1, STEP_SIZE), return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(64, input_shape=(1, STEP_SIZE), return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(16,activation='relu'))        
model.add(Dense(1,activation='linear'))


# MODEL COMPILING AND TRAINING
# model.compile(loss='mean_squared_error', optimizer='adagrad') # Try SGD, adam, adagrad and compare!!!
model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
model.fit(trainX, trainY, epochs=1, batch_size=120, verbose=2)

model.save(f"./files/models/{ASSET_NAME.replace('/', '').replace('-', '')}")  

# model = keras.models.load_model(f"./files/models/{ASSET_NAME.replace('/', '').replace('-', '')}")
