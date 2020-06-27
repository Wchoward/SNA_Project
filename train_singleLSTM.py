import pandas as pd
import os
import numpy as np
from math import cos, sin, atan2, sqrt, pi, radians, degrees, asin
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

scaler = MinMaxScaler(feature_range=(0, 1))
os.environ['CUDA_VISIBLE_DEVICES'] = "6"


def parse(x):
    x = x[:4] + ' ' + x[4:6] + ' '+x[6:]
    return datetime.strptime(x, '%Y %m %d')


def read_data(file):
    return pd.read_csv(file, parse_dates=['date'], date_parser=parse)


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def process(cdf):
    v = []
    for i in range(cdf.region.max()+1):
        df = cdf[cdf.region == i].copy()
        df = df.set_index("date")
        df.drop(["city", "region"], axis=1, inplace=True)
        values = df.values
        values = values.astype('float32')
        # scaled = scaler.fit_transform(values)
        reframed = series_to_supervised(values, n_step, 1)
        # reframed.drop(reframed.columns[range(53, 104)], axis=1, inplace=True)
        v.append(reframed.values)
        a = v[0]
        for i in range(1, len(v)):
            a = np.concatenate((a, v[i]), axis=0)
    return a


def process_last_day(cdf):
    v = []
    for i in range(cdf.region.max()+1):
        df = cdf[cdf.region == i].copy()
        df = df.set_index("date")
        df.drop(["city", "region"], axis=1, inplace=True)
        values = df.values
        values = values.astype('float32')
        # scaled = scaler.fit_transform(values)
        reframed = series_to_supervised(values, n_step, 1)
        v.append(np.array([reframed.values[-1]])[:, n_ob:])
    return v


def predict(day_before):
    day_before = np.reshape(
        day_before, (day_before.shape[0], n_step, n_features))
    day_after = model.predict(day_before)
    return day_after


def generate_future_prediction(lastday):
    res_city_lst = []
    for region_begin in lastday:
        res_date_lst = [predict(region_begin)]
        for i in range(1, 30):
            res_date_lst.append(predict(res_date_lst[-1]))
        res_city_lst.append(res_date_lst)
    return res_city_lst


def generate_res_infection_lst(cities_prediction):
    res_infection = []
    res_region = []
    for city_prediction in cities_prediction:
        region_id = 0
        for region in city_prediction:
            for date in region:
                res_infection.append(int(round(date[0][0])))
                res_region.append(region_id)
            region_id = region_id + 1
    return res_infection, res_region


def check_minus(infection):
    for i in range(len(infection)):
        if infection[i] < 0:
            if infection[i - 1] < 6:
                infection[i] = 0
            else:
                infection[i] = int(infection[i - 1])
    return infection


if __name__ == "__main__":
    A = read_data('data/A.csv')
    B = read_data('data/B.csv')
    C = read_data('data/C.csv')
    D = read_data('data/D.csv')
    E = read_data('data/E.csv')

    n_step = 1
    n_features = A.shape[1] - 3
    n_ob = n_step * n_features

    # A1 = read_data('data/A1.csv')
    a = process(A)
    b = process(B)
    c = process(C)
    d = process(D)
    e = process(E)
    data = np.concatenate((a, b, c, d, e), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        data[:, :n_ob], data[:, n_ob:], test_size=0.2, random_state=2)

    X_train = np.reshape(X_train, (X_train.shape[0], n_step, n_features))
    X_test = np.reshape(X_test, (X_test.shape[0], n_step, n_features))

    model = Sequential()
    model.add(LSTM(units=30, input_shape=(X_test.shape[1], X_test.shape[2])))
    # model.add(LSTM(units=52))
    model.add(Dense(n_features))
    # model.add(Dense(1))

    model.compile(loss='msle', optimizer='adam')
    model.fit(X_train, y_train, epochs=10, batch_size=16,
              validation_data=(X_test, y_test), verbose=2)

    # Evaluation
    yhat = model.predict(X_test)

    # RMSE
    rmse = sqrt(mean_squared_error(y_test, yhat))
    print('Test RMSE: %.3f' % rmse)

    a_lastday = process_last_day(A)
    b_lastday = process_last_day(B)
    c_lastday = process_last_day(C)
    d_lastday = process_last_day(D)
    e_lastday = process_last_day(E)

    a_prediction = generate_future_prediction(a_lastday)
    b_prediction = generate_future_prediction(b_lastday)
    c_prediction = generate_future_prediction(c_lastday)
    d_prediction = generate_future_prediction(d_lastday)
    e_prediction = generate_future_prediction(e_lastday)

    infection, region = generate_res_infection_lst(
        [a_prediction, b_prediction, c_prediction, d_prediction, e_prediction])
    infection = check_minus(infection)
    submission = pd.read_csv('data/submission.csv', header=None,
                             names=['city', 'region', 'date', 'infection'])
    submission['infection'] = infection
    submission['region'] = region
    submission.to_csv('data/test_submission_single_LSTM.csv',
                      header=None, index=False)
