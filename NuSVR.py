import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#################################################################
########################### LOAD DATA ###########################
#################################################################

# Load the VIX data
vix = pd.read_csv('/Users/Ilyas/Documents/Mémoire/VIX.csv')
vix.rename(columns={'Date':'date'}, inplace=True)
vix.set_index('date', inplace=True)

vix.dropna(inplace=True)
vix.index = pd.to_datetime(vix.index)

# Load the market data
market_index = pd.read_csv('/Users/Ilyas/Documents/Mémoire/Market Index.csv')
market_index.rename(columns={'caldt': 'date', 'vwretd': 'Market RET'}, inplace=True)
market_index['date'] = pd.to_datetime(market_index['date'])
market_index.set_index('date', inplace=True)

market_index['volatility'] = market_index['Market RET'].rolling(22).std()
market_index = market_index.merge(vix, how='left', left_index=True, right_index=True)
market_index.loc[:, 'vix'] = market_index.ffill()
market_index.dropna(subset=['Market RET', 'volatility', 'vix'], inplace=True)

# Extract end-of-month volatility
end_of_month_vol = market_index['volatility'].resample('ME').last()


#################################################################
######################### PREPARE DATA ##########################
#################################################################

# Function to create lagged features
def create_lagged_features(df, lag=5):
    for i in range(1, lag+1):
        df[f'Market RET_lag_{i}'] = df['Market RET'].shift(i)
        df[f'vix_lag_{i}'] = df['vix'].shift(i)
        df[f'volatility_lag_{i}'] = df['volatility'].shift(i)
    return df

# Create lagged features
market_index = create_lagged_features(market_index, lag=5)
market_index.dropna(inplace=True)
market_index.drop_duplicates(subset=['Market RET', 'volatility', 'vix'],inplace=True)

# Function to have the last trading have the index of the last day of the month
def adjust_last_day_to_eom(df):
    # Group by year and month to find the last data point for each month
    df['month'] = df.index.to_period('M')
    last_per_month = df.groupby('month').tail(1)
    
    # Adjust the index to the end of the month if needed
    adjusted_idx = []
    for original_idx in last_per_month.index:
        eom_idx = original_idx.to_period('M').to_timestamp('M')
        if original_idx != eom_idx:
            adjusted_idx.append((original_idx, eom_idx))
            
    # Apply the adjustments
    for original_idx, eom_idx in adjusted_idx:
        df.loc[eom_idx] = df.loc[original_idx]
        df = df.drop(original_idx)

    df = df.sort_index()  # Ensure the DataFrame is sorted by index
    return df.drop(columns='month')


# Apply the function to modify the index of the last trading data
market_index = adjust_last_day_to_eom(market_index)

# Prepare the dataset
features = ['Market RET', 'vix', 'volatility'] + [f'Market RET_lag_{i}' for i in range(1, 6)] + [f'vix_lag_{i}' for i in range(1, 6)] + [f'volatility_lag_{i}' for i in range(1, 6)]
dataset = market_index[features].values



def create_sequences_with_eom_target(data, vol_end_of_month, time_steps=22):
    sequences = []
    labels = []
    eom_dates = vol_end_of_month.index

    for i in range(len(data) - time_steps):
        current_date = market_index.index[i + time_steps - 1]
        eom_current_month = current_date - pd.offsets.MonthEnd(0)
        eom_next_month = eom_current_month + pd.DateOffset(months=1)
        eom_next_month = eom_next_month - pd.offsets.MonthEnd(1)
        
        if eom_next_month in eom_dates:
            sequences.append(data[i:(i + time_steps)])
            eom_idx = np.where(eom_dates == eom_next_month)[0][0]
            labels.append(vol_end_of_month.iloc[eom_idx])
    
    print(f"Total sequences created: {len(sequences)}")
    return np.array(sequences), np.array(labels)

time_steps = 22
X, y = create_sequences_with_eom_target(dataset, end_of_month_vol, time_steps)
print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

# Flatten the sequences for SVR
X_flattened = X.reshape(X.shape[0], -1)

#################################################################
####################### MODEL DEFINITION ########################
#################################################################

# Model definition with additional LSTM layers and dropout
svr = NuSVR(nu=0.5, C=2, kernel='rbf', gamma=0.001)


#################################################################
#################### OUT-OF-SAMPLE PREDICTION ###################
#################################################################

# Expanding window training and prediction
start_date = '1992-01-31'

# Lists to store the predictions and the actual values
predictions = []
actuals = []

input_shape = (X.shape[1], X.shape[2])

eom_dates = end_of_month_vol.index
eom_start_index = eom_dates.get_loc(start_date)

# Loop over each date to get the predictions
for eom_date in eom_dates[eom_start_index:]:
    
    # Get the index of the data
    loc = market_index.index.get_loc(eom_date)
    if isinstance(loc, slice):
        current_index = loc.start
    else:
        current_index = loc
        
    if current_index + time_steps > len(X):
        break

    # Keep track of the month being processed
    print(f"Processing month: {eom_date.strftime('%Y-%m')}")

    # Get the training and test data
    X_train, X_test = X_flattened[:current_index], X_flattened[current_index:current_index + 1]
    y_train, y_test = y[:current_index], y[current_index:current_index + 1]
    
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    if X_train.shape[0] == 0 or X_test.shape[0] == 0 or y_train.shape[0] == 0 or y_test.shape[0] == 0:
        continue
    
    # Fit the model
    svr.fit(X_train_scaled, y_train_scaled)
    
    # Get the prediction
    pred = svr.predict(X_test_scaled, verbose=0)
    pred_scaled = svr.predict(X_test_scaled)
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))

    
    # Store the results in the list
    predictions.append(pred[0][0])
    actuals.append(y_test[0])

if predictions:
    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)
    
# Define the root mean squared error (RMSE)
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true)**2))
print('RMSE', rmse(actuals,predictions))


# Save predictions
predictions = pd.DataFrame(predictions)   
pred_vol = sqlite3.connect(database="/Users/Ilyas/Documents/Mémoire/pred_vol.sqlite")
predictions.to_sql("MarketVolNuSVR", pred_vol)
    
    
#################################################################
######################### PLOT RESULTS ##########################
#################################################################

# Store the predictions and the actual values in dataframes with the same index
unique_dates = pd.to_datetime(eom_dates)

pred = pd.DataFrame(predictions.iloc[:,0].values, index= unique_dates[27:])
act = pd.DataFrame(actuals, index= unique_dates[27:])
plt.figure(figsize=(10, 6)) 
plt.plot(pred.index, pred, label='Predicted Volatility', c='crimson')
plt.plot(act.index, act, label='Actual Volatility', c='dodgerblue')
plt.grid(color='lightgrey', linestyle='-', linewidth=0.5, zorder=0)
plt.ylabel("Volatility")

# Set the x-axis to start in 1992
plt.xlim(unique_dates[27], unique_dates[-1])
plt.ylim([0,0.06])

# Formatting the x-axis to show dates nicely
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gcf().autofmt_xdate()  # Rotate date labels

# Display the legend
plt.legend()

# Save the figure
plt.savefig("MarketVolNuSVR.png", bbox_inches="tight")
plt.show()
