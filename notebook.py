import sqlite3
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

# FUNCTION TO CREATE 1D DATA INTO TIME SERIES DATASET
def new_dataset(dataset, step_size):
	data_X, data_Y = [], []
	for i in range(len(dataset)-step_size-1):
		a = dataset[i:(i+step_size), 0]
		data_X.append(a)
		data_Y.append(dataset[i + step_size, 0])
	return np.array(data_X), np.array(data_Y)

ASSET_NAME =  "ALTHUOB/BTC-CXDX"
# ASSET_NAME =  "Z-CRY/IDX"
# FOR REPRODUCIBILITY
np.random.seed(7)

# IMPORTING DATASET
con = sqlite3.connect('./data/trade.db')
sql = f"""
    SELECT price_open as Open, price_high as High, price_low as Low, price_close as Close
    FROM asset_prices ap 
    WHERE asset_name = '{ASSET_NAME}'
    --AND created_at > '2021-05-14T17:50:35.000000Z'
    ORDER BY created_at asc
"""
dataset = pd.read_sql_query(sql, con)

# CREATING OWN INDEX FOR FLEXIBILITY
obs = np.arange(1, len(dataset) + 1, 1)


# TAKING DIFFERENT INDICATORS FOR PREDICTION
OHLC_avg = dataset.mean(axis = 1)
HLC_avg = dataset[['High', 'Low', 'Close']].mean(axis = 1)
close_val = dataset[['Close']]

# PLOTTING ALL INDICATORS IN ONE PLOT
# plt.plot(obs, OHLC_avg, 'r', label = 'OHLC avg')
# plt.plot(obs, HLC_avg, 'b', label = 'HLC avg')
# plt.plot(obs, close_val, 'g', label = 'Closing price')
# plt.legend(loc = 'upper right')
# plt.show()


# PREPARATION OF TIME SERIES DATASE
OHLC_avg = np.reshape(OHLC_avg.values, (len(OHLC_avg),1)) # 1664
scaler = MinMaxScaler(feature_range=(0, 1))
OHLC_avg = scaler.fit_transform(OHLC_avg)


# TRAIN-TEST SPLIT
train_OHLC = int(len(OHLC_avg) * 0.99)
test_OHLC = len(OHLC_avg) - train_OHLC
# print(train_OHLC, test_OHLC)
train_OHLC, test_OHLC = OHLC_avg[0:train_OHLC,:], OHLC_avg[train_OHLC:len(OHLC_avg),:]

# TIME-SERIES DATASET (FOR TIME T, VALUES FOR TIME T+1)
STEP_SIZE = 1000
trainX, trainY = new_dataset(train_OHLC, STEP_SIZE)
testX, testY = new_dataset(test_OHLC, STEP_SIZE)

# RESHAPING TRAIN AND TEST DATA
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# LSTM MODEL
# model = Sequential()
# model.add(LSTM(256, input_shape=(1, STEP_SIZE), return_sequences = True))
# model.add(LSTM(128))
# model.add(Dense(1))
# model.add(Activation('linear'))

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

model.save("./files/model.json")    

# PREDICTION
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# FUTURE PREDICTION
last30 = OHLC_avg[-STEP_SIZE:]
last30 = list(np.reshape(last30, (last30.shape[0])))

FUTURE_PREDICTION_SIZE = 1000
future_predictions = []
for x in range(FUTURE_PREDICTION_SIZE):
    last30_reshaped = np.reshape(last30, (1, 1 ,len(last30)))
    future_prediction = model.predict(last30_reshaped)
    future_predictions.append(future_prediction[0][0])
    last30.append(future_prediction[0][0])
    last30.pop(0)

future_predictions_reshaped = np.reshape(future_predictions, (len(future_predictions), 1))
# exit(0)


# DE-NORMALIZING FOR PLOTTING
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

future_predictions_reshaped = scaler.inverse_transform(future_predictions_reshaped)


# TRAINING RMSE
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train RMSE: %.2f' % (trainScore))


# TEST RMSE
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test RMSE: %.2f' % (testScore))

# CREATING SIMILAR DATASET TO PLOT TRAINING PREDICTIONS
trainPredictPlot = np.empty_like(OHLC_avg)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[STEP_SIZE:len(trainPredict)+STEP_SIZE, :] = trainPredict


# CREATING SIMILAR DATASSET TO PLOT TEST PREDICTIONS
testPredictPlot = np.empty_like(OHLC_avg)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(STEP_SIZE*2)+1:len(OHLC_avg)-1, :] = testPredict

# print(testPredictPlot.shape)
# print(testPredictPlot)
# exit(0)

futurePredictPlot = np.ndarray((OHLC_avg.shape[0] + FUTURE_PREDICTION_SIZE, 1))
futurePredictPlot[:, :] = np.nan
futurePredictPlot[OHLC_avg.shape[0]:OHLC_avg.shape[0] + FUTURE_PREDICTION_SIZE, :] = future_predictions_reshaped
# print(futurePredictPlot.shape)
# print(futurePredictPlot)
# exit(0)


# DE-NORMALIZING MAIN DATASET 
OHLC_avg = scaler.inverse_transform(OHLC_avg)


# PLOT OF MAIN OHLC VALUES, TRAIN PREDICTIONS AND TEST PREDICTIONS
plt.plot(OHLC_avg, 'g', label = 'original dataset')
plt.plot(trainPredictPlot, 'r', label = 'training set')
plt.plot(testPredictPlot, 'b', label = 'predicted stock price/test set')
plt.plot(futurePredictPlot, 'y', label = 'future stock price/test set')
# plt.legend(loc = 'upper right')
plt.xlabel('Time in Days')
plt.ylabel('OHLC Value of Apple Stocks')
plt.show()


# PREDICT FUTURE VALUES
# last_val = testPredict[-1]
# last_val_scaled = last_val/last_val
# next_val = model.predict(np.reshape(last_val_scaled, (1,1,1)))
# print("Last Day Value:", np.asscalar(last_val))
# print("Next Day Value:", np.asscalar(last_val*next_val))
# print(np.append(last_val, next_val))

