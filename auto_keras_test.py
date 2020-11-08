import autokeras as ak
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import IPython

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


# Loading preprocessed data
base_path = 'EA-LSTM/Dataset/simulation_data/'
dataset = pd.read_csv(base_path + 'simulation_data_v2_latent_state.csv', header=0, index_col=0)
dataset.dropna(inplace = True, axis = 0)
dataset.set_index("Timestamp", inplace = True)
dataset.drop(["Date"], inplace = True, axis = 1)

# load values
encoder = LabelEncoder()
dataset["Day of Week"] = encoder.fit_transform(dataset["Day of Week"])
values = dataset.values

# rearrange to put "energy" last 
cols = dataset.columns.tolist()
new_cols = list([x for x in cols if x!='Energy']) + ["Energy"]
dataset = dataset[new_cols]

# saving the dataset 
dataset.to_csv(base_path + 'simulation_data_v2_touched.csv')

# Unified data format as float32
values = values.astype('float32')

# Normalized
scaler = MinMaxScaler(feature_range = (0, 1))
scaled = scaler.fit_transform(values)

scaler_Y = MinMaxScaler(feature_range = (0, 1))

# Sliding window
reframed = series_to_supervised(data = scaled, col_names = dataset.columns, n_in = 18, n_out = 1)

n_train_hours = int(reframed.shape[0] * 0.8)
train = reframed.iloc[:n_train_hours, :]
valid = reframed.iloc[n_train_hours:16779, :]

column_names = reframed.columns

column_names = column_names.drop('Energy(t)')
data_type = (len(column_names)) * ['numerical'] 
data_type = dict(zip(column_names, data_type))

# IPython.embed()

regressor = ak.StructuredDataRegressor(max_trials=2, column_types=data_type)
regressor.fit(x=train.drop(columns=['Energy(t)']), y=train['Energy(t)'])

IPython.embed()

