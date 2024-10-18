import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from flask import Flask, jsonify

app = Flask(__name__)

# Fetch Bitcoin data from Binance
def get_binance_data(symbol='BTCUSDT', interval='1d', limit=500):
    url = f'https://api.binance.com/api/v3/klines'
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                                     'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
                                     'Taker buy quote asset volume', 'Ignore'])
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df['Close'] = df['Close'].astype(float)
    return df[['Open time', 'Close']]

# Function to build and train the model
def train_model():
    btc_data = get_binance_data()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(btc_data['Close'].values.reshape(-1, 1))

    lookback = 60
    X_train, y_train = [], []

    for i in range(lookback, len(scaled_data)):
        X_train.append(scaled_data[i-lookback:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    predicted_prices = model.predict(X_train)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    return btc_data['Open time'][lookback:], btc_data['Close'][lookback:], predicted_prices

# Define a route to serve predictions
@app.route('/')
def index():
    times, actual_prices, predicted_prices = train_model()

    # Prepare the result as a JSON response
    result = {
        "times": times.strftime('%Y-%m-%d').tolist(),
        "actual_prices": actual_prices.tolist(),
        "predicted_prices": predicted_prices.flatten().tolist()
    }
    return jsonify(result)

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
