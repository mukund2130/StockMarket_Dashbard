# train_model.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Dummy training data (replace with real stock data for production use)
x_train = np.random.rand(100, 60, 1)
y_train = np.random.rand(100, 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# Compile and train
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Save model
model.save("lstm_model.h5")
print("✅ Model trained and saved as lstm_model.h5")
