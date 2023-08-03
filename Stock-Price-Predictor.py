#1#########################################################################################################################
# Author: Safwan Alam, Nadeem Hemani, Nabeeha Ashfaq, Micheal Joseph                                                      #
# Description: The purpose of this program is to use neural networks called Long Short Term Memory (LSTM) in hopes        #
# to predict stock prices for a given corporation in this example we will be attempting to predict (Microsoft Inc 'MSFT') #
# by seeding stock data from yahoo.                                                                                       #
##########################################################################################################################

#2
import math
import numpy as np
import pandas as pd
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#3Fetch stock data: start = 06/29/2010 and end = to today's date.
currDate = datetime.today().strftime('%Y-%m-%d')
dataFrame = web.DataReader('MSFT', data_source='yahoo', start='2010-06-29', end=currDate)
dataFrame

#4 returns dataset count (rows, columns)
dataFrame.shape

#5 graph closing price history
plt.figure(figsize=(16, 8))
plt.title('Microsoft - Closing Price History')
plt.plot(dataFrame['Close'])
plt.xlabel('Years', fontsize=20)
plt.ylabel('Close Price in USD ($)', fontsize=20)
plt.show()

#6 Filter stock data from dataFrame with only the 'Close' col
data = dataFrame.filter(['Close'])
dataset = data.values
# Setup training variables
td_len = math.ceil(len(dataset) * .8)

td_len

#7 preprocessing data transformation
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

scaled_data

#8 dataset training creation
train_data = scaled_data[0:td_len, :]
# split the data for independent training variables (x) and dependent training variables (y)
x_train = []
y_train = []

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])
  if i<= 61:
    print(x_train)
    print(y_train)
    print()

#9 convert to numpy array
x_train = np.array(x_train)
y_train = np.array(y_train)

#10 reshape for 3d since LSTM expects 3d
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

#11 build models for LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#12 compile model
model.compile(optimizer='adam', loss='mean_squared_error')

#13 Model training
model.fit(x_train, y_train, batch_size=1, epochs=1)

#14 Dataset for testing set
test_data = scaled_data[td_len - 60 : , :]
x_test = []
y_test = dataset[td_len:, :]

for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i,0])


#15 convert to numpy array
x_test = np.array(x_test)

#16 data reshape
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#17 models for predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#18 evaluate the model using Root Mean Squared Error(RMSE)
rmse = np.sqrt(np.mean( predictions - y_test)**2)
rmse

#19 graph data
subtitle = "RMSE = " + str(rmse) + ", 0 is ideal."
train = data[:td_len]
valid = data[td_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title(subtitle , fontsize=10)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Close Price in USD ($)', fontsize=16)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Raw', 'Test', 'Predictions'], loc='upper left')
plt.suptitle('Microsoft - Predictions', fontsize=20)
plt.show()

#20 show comparison between valid and prediction prices
valid

#21 trying to predicit Microsoft stock for specific date
msft = web.DataReader('MSFT', data_source='yahoo', start='2010-06-29', end='2021-04-09')
# new dataFrame
new_dataFrame = msft.filter(['Close'])
last_60_days = new_dataFrame[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print("Predicted price: " + str(pred_price))

#22 trying to predicit MSFT stock for specific date

msft2 = web.DataReader('MSFT', data_source='yahoo', start='2021-04-09', end='2021-04-09')
print("Actual Closing price: " + str(msft2['Close'][0]))
