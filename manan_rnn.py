#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt

import IPython

# %%
# df = pd.read_csv(
#     "./new_action_energy_data/0_extra_train/more_data_action_energy_linear_extratrain_0.csv"
# )

df = pd.read_csv(
    "simulation_data_v2.csv"
    )

df.dropna(inplace = True, axis = 0)
df.set_index("Timestamp", inplace = True)
df.drop(["Date"], inplace = True, axis = 1)

# # %%
# cbh = pd.offsets.CustomBusinessHour(start="08:00", end="18:00")
# df["Timestamp"] = pd.date_range(
#     start=pd.Timestamp("2018-09-20T08"), freq=cbh, periods=len(df)
# )


# %%
""" Reference: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/ """


### Manan -- I rewrote this so that it returns more informative column names. 
## use col_names = dataset.columns as an input where necessary

def series_to_supervised(data, col_names, n_in = 1, n_out = 1, dropnan = True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    
    #(t-n, ... t-1) --> i.e. steps into the past 
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (col, i)) for col in col_names]
    
    #(t, t+1, ... t+n) --> i.e. steps into the future 
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('%s(t)' % (col)) for col in col_names]
        else:
            names += [('%s(t + %d)' % (col, i)) for col in col_names]
    
    # concat
    agg = pd.concat(cols, axis = 1)
    agg.columns = names
    
    # dropnan
    if dropnan:
        agg.dropna(inplace = True)
    return agg


# %%

# rearrange to put "energy" last 
cols = df.columns.tolist()
new_cols = list([x for x in cols if x!='Energy']) + ["Energy"]
df = df[new_cols]

# Categorical variable processing
encoder = LabelEncoder()
df["Day of Week"] = encoder.fit_transform(df["Day of Week"])

df_y = df["Energy"]
df_X = df.drop("Energy", axis = 1)

# rewriting so that we have two MinMaxScalers... otherwise hard to deal with afterwards 
values_y = df_y.values
values_X = df_X.values
scaler_X = MinMaxScaler(feature_range = (0, 1))
scaler_y = MinMaxScaler(feature_range = (0, 1))

values_y = values_y.astype("float32")
values_X = values_X.astype("float32")

scaled_X = scaler_X.fit_transform(values_X)
scaled_y = scaler_y.fit_transform(values_y.reshape(-1, 1))

scaled = np.concatenate((scaled_X, scaled_y), axis = 1)

reframed = series_to_supervised(data = scaled, col_names = df.columns.tolist(), n_in = 18, n_out = 1)



# %%
""" Let's do a 70:30 training test split """
values_reframed = reframed.values
train_margin = int(len(values_reframed) * 0.7)
train = values_reframed[:train_margin, :]
test = values_reframed[train_margin:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# IPython.embed()

# %%
# design network
model = Sequential()
model.add(LSTM(128, input_shape = (train_X.shape[1], train_X.shape[2]), return_sequences = True))
model.add(LSTM(128, return_sequences = False))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam")
# fit network
history = model.fit(
    train_X,
    train_y,
    epochs=50,
    batch_size=256,
    validation_data=(test_X, test_y),
    verbose=1,
    shuffle=False,
)

# # plot history
# plt.plot(history.history["loss"], label="train")
# plt.plot(history.history["val_loss"], label="test")
# plt.legend()
# plt.show()


# %%
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# %%
# invert scaling for forecast
inv_yhat = scaler_y.inverse_transform(yhat)
inv_yhat = inv_yhat[:, -1]

#%%
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = scaler_y.inverse_transform(test_y)
inv_y = inv_y[:, -1]

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print("Test RMSE: %.3f" % rmse)
print("Predicted y")
print(list(inv_yhat))
print("Y:")
print(list(inv_y))
# %%
