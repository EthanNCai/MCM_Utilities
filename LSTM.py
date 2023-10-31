from numpy import array
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def turn_into_seq_sample(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# ========== configuration =============

# 1. define how many past data points you want to reference to predict a new one (aka time steps)
time_steps = 15

# 2. define how many future data points you want to predict
predict_steps = 3

# 3. your file name
file_name = 'samples.xlsx'

# 4. define where your dataset is located in your excel file (aka column)
target_column = 0

# ======================================


# Read the file and turn it into a Python list
data = pd.read_excel(file_name)
data = data.iloc[:, target_column].tolist()

data_original = data

# split into sequential samples
X, y = turn_into_seq_sample(data, time_steps)

# reshape from [samples, time_steps] into [samples, time_steps, 1]
X = X.reshape((X.shape[0], X.shape[1], 1))

# define the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(time_steps, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# train the model
model.fit(X, y, epochs=200, verbose=0)


for i in range(predict_steps):
    # prepare input data for prediction
    input_data = np.array(data[-time_steps:])
    input_data = np.expand_dims(input_data, axis=-1)
    input_data = np.expand_dims(input_data, axis=0)

    # predict the future values
    predicted = model.predict(input_data)
    data.append(float(predicted[0][0]))
    

# predicted
print('predicted',data[-predict_steps:])