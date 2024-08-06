import numpy as np
import pandas as pd
import sqlite3
import statsmodels.api as sm
from pandas.tseries.offsets import MonthEnd

#################################################################
########################### LOAD DATA ###########################
#################################################################

# Access the database
crsp_compustat = sqlite3.connect(database="/Users/Ilyas/Documents/Mémoire/crsp_compustat.sqlite")

# Import the CRSP monthly data
crspd =  pd.read_sql_query(
                            sql="SELECT * FROM crsp_d",
                            con=crsp_compustat    
                            )

# Rename the columns
crspd.rename(columns={'permno':'PERMNO','bidlo':'LOW','askhi':'HIGH','prc':'CLOSE'}, inplace=True)

# Dictionary to type32
linktotype = {'PERMNO': int,
              'LOW': float,
              'HIGH': float,
              'CLOSE': float}

# Set the type of the variables
crspd = crspd.astype(linktotype)

# Create a month column and get the list years
crspd['date'] = pd.to_datetime(crspd['date'], errors='coerce')

# Get the list of PERMNOs:
PERMNOs = crspd['PERMNO'].unique().tolist() 

# Compute the log high, low and close prices
crspd.loc[:,'HIGH'] = np.log(crspd.loc[:,'HIGH'].abs())
crspd.loc[:,'LOW'] = np.log(crspd.loc[:,'LOW'].abs())
crspd.loc[:,'CLOSE'] = np.log(crspd.loc[:,'CLOSE'].abs())

# Create the mid-price column
crspd['MID'] = (crspd['HIGH'] + crspd['LOW']) / 2

# Set the date and the PERMNO as index
#crspd.set_index('PERMNO', inplace=True)
crspd.set_index(['PERMNO','date'], inplace=True)

# Delete unused columns
crspd.drop(columns=['LOW', 'HIGH'], inplace=True)

#################################################################
################### COMPUTE TRANSACTION COSTS ###################
#################################################################

# Create an empty dataframe that will hold the transaction costs of every stock
t_costs = pd.DataFrame(index=pd.date_range(start='1966-01-01',end='2022-12-31', freq="ME"))


# Loop over every stock
for PERMNO in PERMNOs:
    
    # Get the data of the stock PERMNO
    data = crspd.loc[PERMNO].copy()
    
    # Set the dates as the index
    data.reset_index(inplace=True)
    data.set_index('date', inplace=True)
    
    # Create the shifted (d+1) close and high log prices
    data['shift_MID'] = data['MID'].shift(-1)
    data['shift_CLOSE'] = data['CLOSE'].shift(-1)
    
    # Compute the two-day squared bid-ask spread
    data['squared_spread'] = (4 * (data['CLOSE'] - data['MID']) * (data['shift_CLOSE'] - data['shift_MID'])).clip(lower=0)

    
    # Compute the two-day bid-ask spread
    data['spread'] = np.sqrt(data['squared_spread'])
    
    # Get the mean two-day bid-ask spread per month
    avg_spread = data['spread'].resample('ME').mean()
    
    # Add the average to the dataframe
    t_costs = t_costs.merge(avg_spread.rename(PERMNO), how='left', left_index=True, right_index=True)
    



# Save to SQL
n_splits = 50 # Split the columns into 50 parts
columns_split = np.array_split(t_costs.columns, n_splits)

# Save each part to the SQL database
for i, cols in enumerate(columns_split, 1):
    part_df = t_costs[cols]
    part_df.to_sql(name=f't_costs_filled_part_{i}', 
              con=crsp_compustat, 
              if_exists="replace",
              index=True) 



#################################################################
############### COMPUTE MISSING TRANSACTION COSTS ###############
#################################################################


##  Compute the idiosyncratic volatility

# Load Market data
market_returns = pd.read_csv('/Users/Ilyas/Documents/Mémoire/Market Index.csv')
market_returns.rename(columns={'caldt':'date', 'vwretd':'Market RET'}, inplace=True)
market_returns['date'] = pd.to_datetime(market_returns['date'])

# Load the risk-free rate data from French's website
FF_data = pd.read_csv("/Users/Ilyas/Documents/Mémoire/F-F_Research_Data_Factors_daily.csv", skiprows=3, nrows=25732)
FF_data.rename(columns={"Unnamed: 0":"date"}, inplace="True")
FF_data["date"] = pd.to_datetime(FF_data["date"], format="%Y%m%d")

FF_data.RF = FF_data.RF/100

#Load stock data
crspd = pd.read_sql_query(
    sql="SELECT * FROM t_costs_crspd",
    con=crsp_compustat
)

crspd['date'] = pd.to_datetime(crspd['date'])

# Compute the daily returns for each stock
crspd['daily_return'] = crspd.groupby('PERMNO')['CLOSE'].pct_change(fill_method=None)

# Add the market data and the risk-free rate to the crspd dataframe
crspd = crspd.merge(market_returns, how='left', on='date')
crspd = crspd.merge(FF_data[['date', 'RF']], how='left', on='date')

# Compute the market's excess returns
crspd['Excess Market return'] = crspd['Market RET'] - crspd['RF']
crspd = crspd.dropna(subset=['Excess Market return'])

# Sort stocks by PERMNO and date
crspd = crspd.sort_values(['PERMNO','date'])


def rolling_std_residuals(group, window=63, min_periods=60):
    """
    Calculate the rolling standard deviation of residuals from a regression of daily returns on excess market returns.
    
    Parameters
    ----------
    group : Pandas DataFrame
        DataFrame containing the daily returns and excess market returns for a group of stocks.
    window : int, optional
        The size of the rolling window (default is 63).
    min_periods : int, optional
        Minimum number of observations in the window required to have a value (default is 60).
        
    Returns
    -------
    Pandas Series
        Series containing the rolling standard deviation of the residuals.
    """
    
    residuals_std = []  # Initialize a list to store the rolling standard deviations of the residuals
    
    # Loop over each row in the group DataFrame
    for i in range(len(group)):
        # If the number of periods is less than the minimum required, append NaN to the results
        if i + 1 < min_periods:
            residuals_std.append(np.nan)
        else:
            # Select the subset of data within the rolling window
            subset = group.iloc[max(0, i+1-window):i+1]
            
            # If there are any missing values in the required columns, append NaN to the results
            if subset['Excess Market return'].isnull().sum() > 0 or subset['daily_return'].isnull().sum() > 0:
                residuals_std.append(np.nan)
                continue

            # Set up the regression model with a constant and the excess market returns
            X = sm.add_constant(subset['Excess Market return'])
            y = subset['daily_return']

            try:
                # Fit the OLS regression model
                model = sm.OLS(y, X).fit()
                # Get the residuals from the model
                residuals = model.resid
                # Calculate the standard deviation of the residuals and append to the results
                residuals_std.append(np.std(residuals))
            except:
                # If there's an error in the regression process, append NaN to the results
                residuals_std.append(np.nan)
    
    # Return the results as a Pandas Series with the same index as the input group DataFrame
    return pd.Series(residuals_std, index=group.index)

# Apply the rolling calculation for each group of 'PERMNO'
crspd['std_residuals'] = crspd.groupby('PERMNO').apply(
    lambda x: rolling_std_residuals(x, window=63, min_periods=60)
).reset_index(level=0, drop=True)



# Compute the monthly Market Equity

# Grouping by stock, year, and month, and selecting the last trading day of each month for each stock
last_trading_days = crspd.groupby(['PERMNO', crspd['date'].dt.year, crspd['date'].dt.month]).apply(lambda x: x.iloc[-1])
last_trading_days = last_trading_days.rename_axis(['PERMNO2', 'year', 'month']).reset_index()

last_trading_days.drop(columns=['level_0','index', 'PERMNO2', 'year', 'month', 'vwretx'], inplace=True)

# Save data of the last trading day of each month
last_trading_days.to_sql(name='crspd_last_trading_days', 
          con=crsp_compustat, 
          if_exists="replace") 


# Load monthly data
crspm = pd.read_sql_query(
    sql="SELECT * FROM crsp_m",
    con=crsp_compustat
)
crspm.rename(columns={'permno': 'PERMNO'}, inplace=True)
crspm = crspm.astype({'PERMNO':int})
crspm['date'] = pd.to_datetime(crspm['date'])

# Merge the data with the idiosyncratic volatility data
crspm = pd.merge(crspm,last_trading_days, how='inner', on=['PERMNO','date'])

def load_transaction_costs():
    parts = []
    for i in range(1, 51):  # Assuming 50 parts
        #part_df = pd.read_sql_query(f"SELECT * FROM t_costs_filled_part_{i}", conn, index_col='date')
        part_df = pd.read_sql_query(f"SELECT * FROM t_costs_filled_part_{i}", crsp_compustat,index_col='index')
        parts.append(part_df)
    return pd.concat(parts, axis=1)

# Load transaction costs
transaction_costs = load_transaction_costs()

# Modify the transaction costs dataframe to have the date as indices and PERMNOs as columns
df_reset = transaction_costs.reset_index()
df_long = df_reset.melt(id_vars='index', var_name='PERMNO', value_name='t_cost')
df_long.rename(columns={'index': 'date'}, inplace=True)

t_cost = df_long
t_costs = t_cost
t_costs = t_costs.astype({'PERMNO':int})
t_costs.date = pd.to_datetime(t_costs.date)



# Ensure that price data is positive
crspm['prc'] = crspm['prc'].abs()

# Compute the market equity at the PERMNO level
crspm['ME'] = crspm['prc'] * crspm['shrout']


crsp_combined = crspm


# Delete unused dataframes
del crspd, crspm,df_long

# Delete unused columns
crsp_combined.drop(columns=['shrcd','exchcd', 'ret', 'retx', 'shrout','prc', 'siccd','level_0', 'index','CLOSE','vwretx',
                            'Market RET','RF', 'Excess Market return', 'daily_return'], inplace=True)


# Rank ME and IVOL
crsp_combined['rank_ME'] = crsp_combined.groupby('date')['ME'].rank()
crsp_combined['rank_IVOL'] = crsp_combined.groupby('date')['std_residuals'].rank()


# Process each date individually
unique_dates = t_costs['date'].unique()

# Initialize the result DataFrame
result_df = pd.DataFrame()

crsp_combined['date'] = crsp_combined['date'] + MonthEnd(0)


# Find the closest match of each stock with missing TCs for each date
for date in unique_dates:
    # Filter data for the specific date
    t_costs_date = t_costs[t_costs['date'] == date].copy()
    crsp_combined_date = crsp_combined[crsp_combined['date'] == date].copy()

    # Merge data for the specific date
    merged = pd.merge(t_costs_date, crsp_combined_date, on=['PERMNO', 'date'], how='left')


    # Drop unused columns
    merged.drop(columns=[ 'ME', 'std_residuals', 'spread'], inplace=True)

    # Calculate Euclidean distance for each missing spread
    def calculate_closest_match(group):
        """
        Fill missing transaction cost ('t_cost') values by finding the closest match based on 
        'rank_ME' and 'rank_IVOL' values within the group.
    
        Parameters
        ----------
        group : Pandas DataFrame
            DataFrame containing columns 'rank_ME', 'rank_IVOL', and 't_cost'.
    
        Returns
        -------
        group : Pandas DataFrame
            DataFrame with missing 't_cost' values filled based on the closest match.
        """
        
        # Identify rows with missing and present 't_cost' values
        missing_spread = group['t_cost'].isnull()
        present_spread = group['t_cost'].notnull()
    
        # If there are no missing 't_cost' values, return the group as is
        if missing_spread.sum() == 0:
            return group
    
        # Extract the rows with present 't_cost' values and relevant columns
        present_data = group.loc[present_spread, ['rank_ME', 'rank_IVOL', 't_cost']]
    
        # Loop over each row with missing 't_cost'
        for idx in group[missing_spread].index:
            row = group.loc[idx]
            
            # Check if both 'rank_ME' and 'rank_IVOL' are not null
            if pd.notnull(row['rank_ME']) and pd.notnull(row['rank_IVOL']):
                # Calculate the Euclidean distance based on both 'rank_ME' and 'rank_IVOL'
                distances = np.sqrt((present_data['rank_ME'] - row['rank_ME']) ** 2 +
                                    (present_data['rank_IVOL'] - row['rank_IVOL']) ** 2)
            elif pd.notnull(row['rank_ME']):
                # Calculate the absolute distance based on 'rank_ME' only
                distances = np.abs(present_data['rank_ME'] - row['rank_ME'])
            else:
                # If 'rank_ME' is null, use the mean 't_cost' from the present data
                group.loc[idx, 't_cost'] = present_data['t_cost'].mean()
                continue
    
            # Find the index of the closest match
            closest_index = distances.idxmin()
            
            # Fill the missing 't_cost' value with the closest match's 't_cost'
            group.loc[idx, 't_cost'] = present_data.loc[closest_index, 't_cost']
    
        # Return the group with filled 't_cost' values
        return group


    # Apply the calculation to the merged data
    filled_t_costs_date = calculate_closest_match(merged)

    # Append the result to the final result DataFrame
    result_df = pd.concat([result_df, filled_t_costs_date])



# Save the intermediate results
result_df.to_sql(name='result_df_20_06', 
          con=crsp_compustat, 
          if_exists="replace",
          index=True) 
    
# Update t_costs with the filled transaction costs
# Merge the result back to the original t_costs DataFrame
t_costs = t_costs.merge(result_df, on=['PERMNO', 'date'], how='left', suffixes=('', '_filled'))

# Use the filled t_cost where original t_cost is NaN
t_costs['t_cost'] = t_costs['t_cost'].combine_first(t_costs['t_cost_filled'])

# Drop the temporary filled column
t_costs.drop(columns=['t_cost_filled'], inplace=True)

# Clean up placeholder columns
t_costs.drop(['rank_ME', 'rank_IVOL'], axis=1, inplace=True)


# Calculate the cross-sectional average of 't_cost' by date
average_t_costs = t_costs.groupby('date')['t_cost'].transform('mean')

# Fill missing values in 't_cost' with the cross-sectional average by date
t_costs['t_cost'].fillna(average_t_costs, inplace=True)

transaction_costs = t_costs.pivot(index='date', columns='PERMNO', values='t_cost')

# Save to SQL
n_splits = 50
columns_split = np.array_split(transaction_costs.columns, n_splits)

# Save each part to the SQL database
for i, cols in enumerate(columns_split, 1):
    part_df = transaction_costs[cols]
    part_df.to_sql(name=f't_costs_filled_20_06_part_{i}', 
              con=crsp_compustat, 
              if_exists="replace",
              index=True) 
    