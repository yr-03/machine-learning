import yfinance as yf

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler

from stock_model import StockLSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = False

ticker = 'MSFT'
df = yf.download(ticker, '2020-01-01')
df['Return'] = df['Close'].pct_change()
df = df.dropna().copy()

seq_length = 100
components = 7 
pred_days = 5

fourier_features = np.zeros(len(df))
for i in range(seq_length, len(df)):
    window_data = df['Return'][i-seq_length:i]
    fft_coefs = np.fft.fft(window_data.values.flatten())

    fft_filtered = fft_coefs.copy()
    if components > 0:
        fft_filtered[components:-components] = 0

    reconstructed = np.fft.ifft(fft_filtered).real
    fourier_features[i] = reconstructed[-1]

df['fourier_cleaned'] = fourier_features
df = df.iloc[seq_length:].copy()

train_ratio = 0.8
split_idx = int(len(df) * train_ratio)

train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

feature_cols = ['Return', 'fourier_cleaned']

new_scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_train = new_scaler.fit_transform(train_df[feature_cols].values)
scaled_test = new_scaler.transform(test_df[feature_cols].values)

def create_sequences(data, seq_length, pred_days):
    X, y = [], []
    for i in range(len(data) - seq_length - pred_days + 1):
        X.append(data[i:i+seq_length, :])
        y.append(data[i+seq_length:i+seq_length+pred_days, 0])
    return np.array(X), np.array(y)

X_train_np, y_train_np = create_sequences(scaled_train, seq_length, pred_days)
X_test_np, y_test_np = create_sequences(scaled_test, seq_length, pred_days)

# Convert to Tensors
X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train_np, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test_np, dtype=torch.float32).to(device)

model = StockLSTM(input_size=2, hidden_size=128, num_layers=2, output_size=5).to(device)
loss_function = nn.MSELoss()
optimiser = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(300):
    y_train_pred = model(X_train)

    loss = loss_function(y_train_pred, y_train)

    if (epoch + 1) % 25 == 0:
        print(f"Epoch {epoch + 1} loss: {loss}")

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

model.eval()

y_test_pred = model(X_test)

y_train_pred_np = y_train_pred.detach().cpu().numpy()
y_train_np = y_train.detach().cpu().numpy()
y_test_pred_np = y_test_pred.detach().cpu().numpy()
y_test_np = y_test.detach().cpu().numpy()

def inverse_transform_target(scaler, multi_day_array):
    rows, days = multi_day_array.shape
    inverted_output = np.zeros((rows, days))
    
    for d in range(days):
        dummy = np.zeros((rows, 2))
        dummy[:, 0] = multi_day_array[:, d]
        dummy_inverted = scaler.inverse_transform(dummy)
        inverted_output[:, d] = dummy_inverted[:, 0]
    return inverted_output

# 1. Unscale the predictions back into actual return percentages
y_test_pred_real_returns = inverse_transform_target(new_scaler, y_test_pred_np)
y_test_real_returns = inverse_transform_target(new_scaler, y_test_np)

# 2. Reconstruct prices from returns
def reconstruct_prices(actual_prices_series, predicted_returns_matrix):
    num_samples, pred_days = predicted_returns_matrix.shape
    reconstructed_prices = np.zeros((num_samples, pred_days))
    
    for i in range(num_samples):
        # Anchor point: The actual close price on the day before the prediction horizon starts
        current_price = actual_prices_series[i]
        for d in range(pred_days):
            current_price = current_price * (1 + predicted_returns_matrix[i, d])
            reconstructed_prices[i, d] = current_price
    return reconstructed_prices

# Grab the actual raw 'Close' price baselines corresponding to your test sequence starts
# This aligns perfectly with the index position where create_sequences cuts off
base_prices = test_df['Close'].values[seq_length : seq_length + len(y_test_pred_real_returns)].flatten()

# Reconstruct predicted and actual prices
predicted_prices = reconstruct_prices(base_prices, y_test_pred_real_returns)
actual_prices = reconstruct_prices(base_prices, y_test_real_returns)

# 3. Plotting Absolute Prices
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(6, 1)

# Align the index dates for the x-axis
plot_dates = test_df.index[seq_length : seq_length + len(y_test_pred_real_returns)]

# 1-Day Ahead Plot
ax1 = fig.add_subplot(gs[:3, 0])
ax1.plot(plot_dates, actual_prices[:, 0], color='blue', label='Actual Price')
ax1.plot(plot_dates, predicted_prices[:, 0], color='green', label='Predicted Price')
ax1.legend()
plt.title(f"{ticker} Stock Price Prediction 1 Day Ahead (Reconstructed)")
plt.xlabel('Date')
plt.ylabel('Price ($)')

# 4-Days Ahead Plot
ax2 = fig.add_subplot(gs[3:6, 0])
ax2.plot(plot_dates, actual_prices[:, 4], color='blue', label='Actual Price')
ax2.plot(plot_dates, predicted_prices[:, 4], color='green', label='Predicted Price')
ax2.legend()
plt.title(f"{ticker} Stock Price Prediction 4 Days Ahead (Reconstructed)")
plt.xlabel('Date')
plt.ylabel('Price ($)')

plt.tight_layout()
plt.show()