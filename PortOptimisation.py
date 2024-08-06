import sqlite3
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from arch.bootstrap import StationaryBootstrap
from numpy.random import RandomState

#################################################################
########################### LOAD DATA ###########################
#################################################################

def load_data():
    conn = sqlite3.connect("/Users/Ilyas/Documents/Mémoire/crsp_compustat.sqlite")
    
    # Load factor returns and market volatility
    factor_returns = pd.read_sql_query("SELECT * FROM factors", conn, index_col='index')
    market_volatility = pd.read_sql_query("SELECT * FROM market_vol", conn, index_col='date')
    
    # Function to load and concatenate split weights data
    def load_weights(name):
        parts = []
        for i in range(1, 51):
            part_df = pd.read_sql_query(f"SELECT * FROM weights_{name}_part_{i}", conn, index_col='date')
            parts.append(part_df)
        return pd.concat(parts, axis=1)
    
    # Load all factor weights
    #factor_weights_names = ['Market', 'SMB', 'HML', 'RMW', 'CMA','MOM', 'IA', 'ROE', 'BAB']
    factor_weights_names = ['Market', 'SMB', 'HML', 'RMW', 'CMA','MOM', 'BAB'] # Exclude I/A and ROE
    factor_weights = [load_weights(name) for name in factor_weights_names]
    
    # Function to load and concatenate split transaction costs data
    def load_transaction_costs():
        parts = []
        for i in range(1, 51):  # Assuming 50 parts
            part_df = pd.read_sql_query(f"SELECT * FROM t_costs_filled_20_06_part_{i}", conn, index_col='date')
           
            parts.append(part_df)
        return pd.concat(parts, axis=1)
    
    # Load transaction costs
    transaction_costs = load_transaction_costs()
    
    # Function to load and concatenate split stock returns data
    def load_stock_returns():
        parts = []
        for i in range(1, 51):  # Assuming 50 parts
            part_df = pd.read_sql_query(f"SELECT * FROM stock_ret_part_{i}", conn, index_col='date', parse_dates='date')
            parts.append(part_df)
        return pd.concat(parts, axis=1)
    
    # Load transaction costs
    stock_returns = load_stock_returns()
    
    # Load the list of PERMNOs
    PERMNOs = pd.read_sql_query("SELECT * FROM PERMNOs", conn)
    PERMNOs = PERMNOs.astype(int)
    PERMNOs = PERMNOs.iloc[:,0].to_list()
    
    conn.close()
    
    return factor_returns, market_volatility, transaction_costs, factor_weights, stock_returns, PERMNOs

# Load data
factor_returns, market_volatility, transaction_costs, factor_weights, stock_returns, PERMNOs = load_data()

# Exclude I/A and ROE factors
factor_returns.drop(columns=['I/A', 'ROE'], inplace=True)


# Access the database
crsp_compustat = sqlite3.connect(database="/Users/Ilyas/Documents/Mémoire/crsp_compustat.sqlite")

# Import the CRSP monthly data
stock_prices =  pd.read_sql_query(
                            sql="SELECT permno, date, prc from crsp_m",
                            con=crsp_compustat    
                            )

stock_prices.drop_duplicates(subset=['permno','date'],inplace=True)
stock_prices = stock_prices.astype({'permno': int, 'prc':float})

stock_prices = stock_prices.pivot_table(index='date', columns='permno', values='prc')
stock_prices = stock_prices.abs()
stock_prices.dropna(how='all', axis=1)
stock_prices.index = pd.to_datetime(stock_prices.index) + MonthEnd(0)

#################################################################
########################## PREPARE DATA #########################
#################################################################

# Reindex the data so that they all share the same row index

# Get the list of unique dates
unique_dates = pd.date_range(start='1969-11-01', end='2022-12-31', freq="ME")

# Redfinition to get the comparison with the other series
unique_dates = pd.date_range(start='1992-04-01', end='2022-12-31', freq="ME")

def reindex_weights(df, row, column=None, fill=0):
    # Reindex the rows on the list of all dates
    df.index = pd.to_datetime(df.index)
    if fill == 0:
        reindexed_df = df.reindex(index=row)
    else:
        reindexed_df = df.reindex(index=row, method='ffill', limit=fill)
    reindexed_df.rename_axis('date', inplace=True)
    
    # Reindex the columns on the list of all PERMNOs
    if column is not None:
        reindexed_df.columns = reindexed_df.columns.astype(int)
        reindexed_df = reindexed_df.reindex(columns=column)
    
    return reindexed_df

# Reindex the dataframes so that they all share the same row and column indices
for i in range(len(factor_weights)):
    factor_weights[i] = reindex_weights(factor_weights[i], row=unique_dates, column=PERMNOs, fill=0)

transaction_costs = reindex_weights(transaction_costs, column=PERMNOs, row=unique_dates, fill=0)
stock_returns = reindex_weights(stock_returns, column=PERMNOs, row=unique_dates, fill=0)
factor_returns = reindex_weights(factor_returns, row=unique_dates, fill=0)
market_volatility = reindex_weights(market_volatility, row=unique_dates, fill=0)
stock_prices = reindex_weights(stock_prices, column=PERMNOs, row=unique_dates, fill=0)



# Fill with 0 for the matrix multiplications
factor_returns.fillna(0.0, inplace=True)
stock_returns.fillna(0.0, inplace=True)
stock_prices.fillna(0.0, inplace=True)

# Transaction costs are half of the spread
transaction_costs = transaction_costs.div(2)

#################################################################
###################### PRECOMPUTE VARIABLES #####################
#################################################################

# Compute the volatility-managed returns
def precompute_extended_factor_returns(factor_returns, market_volatility):
    
    # Number of factors
    K = factor_returns.shape[1]
    
    #c = 1/(1/market_volatility).mean().values.item() #Constant to rescale the volatility managed portfolios
    
    # Get the volatility as a Pandas Series
    market_volatility = market_volatility.iloc[:,0].shift(1) 
    
    # Initialise the lists that will hold the original factors' returns and volatility-managed factors' returns
    extended_factor_returns = []
    extended_factor_vol_returns = []
    
    # Loop over the factors returns series
    for k in range(K):
        extended_factor_returns.append(factor_returns.iloc[:,k].fillna(0))
        extended_factor_vol_returns.append(factor_returns.iloc[:,k].fillna(0).div(market_volatility, axis=0))
    
    # Combine the returns in a dataframe
    extended_factor_returns = pd.concat(extended_factor_returns, axis=1)
    extended_factor_vol_returns = pd.concat(extended_factor_vol_returns, axis=1)
    
    
    extended_factor_vol_returns.rename(columns={0:'MarketVol',1:'SMBVol',2:'HMLVol',3:'RMWVol',4:'CMAVol',5:'MOMVol', 6:'BABVol'}, inplace=True)
    
    # Concatenate the original and volatility managed factors so that they have the same order
    extended_factor_returns = extended_factor_returns.merge(extended_factor_vol_returns, how='left', left_index=True, right_index=True)
    
    return extended_factor_returns

# Compute r_ext, shape: (T, 2 x K)
extended_factor_returns = precompute_extended_factor_returns(factor_returns, market_volatility)


# Get the sample mean of each factor for the in-sample analysis, shape: (T, 2 x K)
mu_ext = extended_factor_returns.expanding(min_periods=1).mean()


# Desired order of the factors, to facilitate further analysis
desired_order = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM', 'BAB',
                 'MarketVol', 'SMBVol', 'HMLVol', 'RMWVol', 'CMAVol', 'MOMVol','BABVol']

# Function to compute expanding window covariance
def expanding_window_covariance(df, desired_order):
    
    # Initialise the list that will hold the covariance matrixes
    cov_list = []
    
    # Compute the r_ext covariance for the in-sample analysis
    for end in range(1, len(df) + 1):
        # Data up to the end of the in-sample window
        window_df = df.iloc[:end]
        
        if end > 1:  # Ensure there are at least 2 data points
            cov_matrix = window_df.cov()
            cov_matrix['Date'] = window_df.index[-1]
            cov_list.append(cov_matrix)
    
    # Concatenate all covariance matrices
    all_cov = pd.concat(cov_list)
    
    # Create a multi-level index
    all_cov = all_cov.set_index(['Date'], append=True)
    all_cov = all_cov.swaplevel(0, 1)
    all_cov = all_cov.sort_index(level=0)
    
    # Reorder rows and columns
    all_cov = all_cov.loc[(slice(None), desired_order), desired_order]
    
    return all_cov

# Get the sample covariance of each factor for the in-sample analysis, shape: (T, 1), and each cell contains a matrix of shape (2 x K, 2 x K)
sigma_ext = expanding_window_covariance(extended_factor_returns, desired_order)
 

# Function to compute the volatility-managed weights matrices
def precompute_weights(factor_weights, market_volatility):
    # Number of factors
    K = len(factor_weights)
    
    # Get the volatility as a Pandas Series
    market_volatility = market_volatility.iloc[:,0]
    
    # Initialise the lists that will contain the weights for the original and volatility managed factors
    factor_weights_list = []
    factor_weights_vol = []
    
    # Loop over each factor
    for k in range(K):
        # Store the weights, and set the weights of other stocks to 0 for the matrix multiplications
        factor_weights_list.append(factor_weights[k].fillna(0.0))
        factor_weights_vol.append(factor_weights[k].fillna(0.0).div(market_volatility, axis=0))
    
    # Add the factor weights of volatility-managed factors, in the same order as the original factors
    factor_weights_list.extend(factor_weights_vol)
    
    return factor_weights_list

# Compute the volatility-managed weights, shape: (2 x K), and each element inside has a shape (T, N)
factor_weights = precompute_weights(factor_weights, market_volatility)


# Function to compute the extended weight matrix
def create_weights_per_dates(factor_weights):
    
    # Check if all dataframes have the same indices
    if not all(df.index.equals(factor_weights[0].index) and df.columns.equals(factor_weights[0].columns) for df in factor_weights):
        raise ValueError("All dataframes in factor_weights must have the same indices and columns")
    
    # Extract common row indices (dates) and column indices (PERMNOs)
    dates = factor_weights[0].index
    permnos = factor_weights[0].columns
    
    # Initialize a dictionary to store numpy arrays by date
    weights_per_dates_dict = {}
    
    # Iterate over each date
    for date in dates:
        # Create a numpy array for the current date of shape (N, 2 x K)
        weights_array = np.zeros((len(permnos), len(factor_weights)))
        
        # Populate the numpy array with values from factor_weights
        for i, df in enumerate(factor_weights):
            weights_array[:, i] = df.loc[date].values
        
        # Add the numpy array to the dictionary with the date as the key
        weights_per_dates_dict[date] = weights_array
    
    # Convert the dictionary to a pandas Series
    weights_per_dates = pd.Series(weights_per_dates_dict, name='weights_per_dates')
    
    return weights_per_dates

# Compute X_ext for each date, shape: (T, 1), and each cell has a shape (N, 2 x K)
weights_per_dates = create_weights_per_dates(factor_weights)


#################################################################
###################### WEIGHTS OPTIMIZATION #####################
#################################################################


# Objective function to minimize
def objective_function(eta, mu_ext, sigma_ext, Lambda, compounded_weights, X_ext, t, TC, gamma):
    """
    

    Parameters
    ----------
    eta : Numpy array, shape: (2 x K, )
        Factor portfolio weights.
    mu_ext : Numpy array, shape: (1, 2 x K)
        Sample mean of factor returns up to date t.
    sigma_ext : Numpy array, shape: (2 x K, 2 x K)
        Sample covariance of factor returns up to date t.
    Lambda : Numpy array, shape: (N, 1)
        Individual transaction cost at date t.
    compounded_weights : Numpy array, shape: (N, 1)
        Position before the rebalancing.
    X_ext : Numpy array, shape: (N, 2 x K)
        Stock weights associated to each factor.
    t : int, shape: 1
        Number of trading periods.
    TC : float, shape: 1
        Cumulative transaction costs.
    gamma : float, shape: 1
        Risk aversion parameter.

    Returns
    -------
    float
        Opposite of an investor mean-variance utility.

    """
    
    # Reshape eta for matrix multiplications and make it a contiguous array
    eta = eta.reshape(-1, 1)


        
    # Sample portfolio mean, shape: 1
    mu_eta = (mu_ext @ eta).item() # Shapes: (1, 2 x K) @ (2 x K, 1)
    
    # Sample portfolio variance, shape: 1
    eta_sigma_eta = np.transpose(eta) @ sigma_ext @ eta # Shapes: (1, 2 x K) @ (2 x K, 2 x K) @ (2 x K, 1)
    
    if t == 1:
        TC_avg = 0
        #print(f"mu_eta: {mu_eta}, eta_sigma_eta: {eta_sigma_eta}")
    else:
        # Compute the change in weights (∆w), shape: (N, 1)
        change_weights = np.abs(X_ext @ eta - compounded_weights) # Shapes: (N, 2 x K) @ (2 x K, 1) - (N, 1)
        
        # Compute the transaction costs associated with the rebalancing, shape: 1
        TC_eta = np.sum(change_weights * Lambda).item() # Shapes: (N, 1) @ (N, 1)

        # Add the previous transaction costs, shape: 1
        TC_avg = (TC_eta + TC) / (t - 1)
    
        #print(f"mu_eta: {mu_eta}, eta_sigma_eta: {eta_sigma_eta}, TC_eta: {TC_eta}, TC_avg: {TC_avg}")

    return -(mu_eta - TC_avg - 0.5 * gamma * eta_sigma_eta).item() 


# Function to find the weights that minimize the opposite of the utility function
def optimize_eta(mu_ext,sigma_ext, Lambda, compounded_weights, X_ext, gamma, TC, t, wgt_eta):
    """
    

    Parameters
    ----------
    mu_ext : Numpy array, shape: (1, 2 x K)
        Sample mean of factor returns up to date t.
    sigma_ext : Numpy array, shape: (2 x K, 2 x K)
        Sample covariance of factor returns up to date t.
    Lambda : Numpy array, shape: (N, 1)
        Individual transaction cost at date t.
    compounded_weights : Numpy array, shape: (N, 1)
        Position before the rebalancing.
    X_ext : Numpy array, shape: (N, 2 x K)
        Stock weights associated to each factor.
    gamma : float, shape: 1
        Risk aversion parameter.
    TC : float, shape: 1
        Cumulative transaction costs.
    t : int, shape: 1
        Number of trading periods.
    wgt_eta : Numpy array, shape: (1, 2 x K)
        Previous weights.

    Returns
    -------
    Results of the minimization of the objective function.

    """
    
    # # Initial guess (previous month's weights), shape: (2 x K, 1)
    initial_eta = wgt_eta
    
    # Define bounds for each variable to be non-negative
    bounds = tuple((0.001, 999) for _ in range(K))
    #bounds = tuple((0.00001, None) for _ in range(len(wgt_eta))]
    #lb = np.append(1, np.repeat(0, 13))
    #ub = np.append(1, np.repeat(9999, 13))
    
    #constraints = ({'type': 'eq', 'fun': lambda eta: np.sum(eta) - 1})
    # Minimization
    result = minimize(
        objective_function, 
        initial_eta, 
        args=(mu_ext, sigma_ext, Lambda, compounded_weights, X_ext, t, TC, gamma),
        method='SLSQP', 
        bounds=bounds,
        options={'ftol': 0.000000000001}
    )

    #result = minimize(objective_function, initial_eta, args=(mu_ext, sigma_ext, Lambda, compounded_weights, X_ext, t, TC, gamma), method='SLSQP', bounds=bounds,  options={'ftol': 1, 'maxiter': 1000})


    return result




gamma = 5  # Risk-aversion parameter

# Initialisation of the dataframe that will contain the weights of the factors
optimal_eta_df = pd.DataFrame(index=unique_dates, columns=desired_order)

# Initialisation of the dataframe that will contain the portfolio transaction costs
portfolio = pd.DataFrame(index=unique_dates, columns=['Portfolio Value','Transaction Costs'])

# Number of factors
K = len(factor_weights) 

# First guess: equal-weighted, shape: (1, 2 x K)
#wgt_eta = np.array([1.3277229465404619, 0.837389381165301, 1.0460160658429223e-16, 0.0, 4.569037125016756, 0.3472973323564199, 5.4119064118295655e-18	, 1.9278842141570563e-05, 	0.0032872695669896755, 0.017182812730402397, 0.04950084105413666, 1.9972369364017215e-18, 0.005329323296599978,0.0045307431142720634])
wgt_eta = np.ones(K)

# Initialise the cumulative transaction costs
TC = 0

# Initial position, shape: (N, 1)
weights = weights_per_dates.loc[unique_dates[121]] @ wgt_eta.reshape(-1, 1)
weights = weights_per_dates.loc[unique_dates[61]] @ wgt_eta.reshape(-1, 1)

# Initialise the number of months of trading, to compute the average transaction costs
t = 1

# Loop over each month of the sample, with an initial window of 10 years
#for month in unique_dates[121:]:
for month in unique_dates[61:]:   
    # Keep track of the month processed
    print(f"{month} is being processed.")

    
    # Compute the updated weights (w^+ = X_ext_(t-1) • (1 + r_t)), shape: (N, 1)  
    compounded_weights = weights.reshape(-1, 1) * (stock_returns.loc[month].add(1)).values.reshape(-1, 1) # Shapes: (N, 1) • (N, 1)
    #compounded_weights = weights * (1 + stock_returns.loc[month].values.reshape(-1, 1))
    
    # Compute the optimal weights through the minimisation
    optimal_eta = optimize_eta(mu_ext.loc[month].values.reshape(1, -1), sigma_ext.loc[month].values, transaction_costs.loc[month].values.reshape(-1, 1), compounded_weights, weights_per_dates.loc[month], gamma, TC, t, wgt_eta)
                                                                                                                                                                
    # Keep the weights as an initial guess for the next optimisation, shape: (2 x K, )
    wgt_eta = optimal_eta.x
    
    # Store the weights
    optimal_eta_df.loc[month] = wgt_eta
    
    
    # Update the cumulative transaction costs
    change_weights = (weights_per_dates.loc[month] @ wgt_eta.reshape(-1, 1)) - compounded_weights # Shape: (N, 2 x K) @ (2 x K, 1) - (N, 1)
    TC_eta = np.sum(np.abs(change_weights) * transaction_costs.loc[month].values.reshape(-1, 1)) # Shape: sum(|(N, 1) • (N, 1)|) = (1, 1)
    
    # Do not consider the transaction costs to get the first position
    if t == 1:
        TC = 0
    else: # Add the transaction costs associated with the rebalancing to the cumulative transaction costs
        TC += TC_eta.item()
    

    # Store the transaction costs
    portfolio.loc[month,'Transaction Costs'] = TC_eta.item()
    
    # Update the out-of-sample portfolio values
    portfolio_value = np.sum((weights_per_dates.loc[month] @ wgt_eta.reshape(-1, 1)) * stock_prices.loc[month].values.reshape(-1,1))
    print(f'Portfolio Value: {portfolio_value}')
    portfolio.loc[month,'Portfolio Value'] = portfolio_value
    
    # Compute X_ext @ eta, shape: (N, 1)
    weights = weights_per_dates.loc[month] @ wgt_eta.reshape(-1, 1) # Shapes: (N, 2 x K) @ (2 x K, 1)
    
    # Update the month counter
    t += 1
    




#################################################################
###################### DISPLAY THE RESULTS ######################
#################################################################

# Compute the portfolio returns
ret = optimal_eta_df.shift(1).dropna(axis=0).mul(extended_factor_returns)

# Get the returns net of transaction costs
port_ret = ret.dropna().sum(axis=1).sub(portfolio["Transaction Costs"])

cmv_ret = port_ret

# #cumret = port_ret.dropna(axis=0).add(1).cumprod().div(10000)
# conn = sqlite3.connect("/Users/Ilyas/Documents/Mémoire/market.sqlite")
# mret =  pd.read_sql_query(
#                             sql="SELECT * from mret",
#                             con=conn,    
#                             index_col='date')


# mret.index = pd.to_datetime(mret.index)

# # Adjust the returns series so that it has the same volatility as the market
# adjustment_factor = mret.std()/port_ret.std()
# port_ret *= adjustment_factor.values.item()

# # Create the cumulative returns series
# cumret = port_ret.dropna(axis=0).add(1).cumprod().sub(1)
# mcumret = mret.loc[cumret.index[0]:].dropna(axis=0).add(1).cumprod().sub(1)

# # Let the series start at the same point
# cumret += 1 - cumret.iloc[0] # The first return will be 0 on the log-axis
# mcumret += 1 - mcumret.iloc[0]

# # Create the plot
# plt.figure(figsize=(10, 6))
# plt.plot(cumret, c='dodgerblue', label='CMV')  # Replace with your actual label
# plt.plot(mcumret, c='lightgrey', label='Market' )

# # Set y-axis to logarithmic scale
# plt.yscale('log')

# # Adding labels and title
# plt.xlabel('Year')
# plt.ylabel('Cumulative Returns (log scale)')
# plt.title('Cumulative Returns of Multifactor Portfolios')
# plt.legend()

# #plt.savefig("CMVvsMarket.png", bbox_inches="tight")
# # Display the plot
# plt.show()

# # Display the Sharpe ratios, assuming a risk-free rate equal to 0.047
# print(f'The Sharpe ratio of the volatility-managed portfolio is {(port_ret.mean())/port_ret.std()}')
# print(f'The Sharpe ratio of the market portfolio is {(mret.mean())/mret.std()}')



# # Save the results
# conn2 = sqlite3.connect("/Users/Ilyas/Documents/Mémoire/portret.sqlite")
# port_ret.to_sql("CMVreturns", con=conn2, index='date')

########################################################################################################################################################################################
########################################################################################################################################################################################
########################################################################################################################################################################################
########################################################################################################################################################################################

pred_vol = sqlite3.connect(database="/Users/Ilyas/Documents/Mémoire/pred_vol.sqlite")
market_volatility_pred = pd.read_sql_query(
                            sql="SELECT * from MarketVol1Cell1002",
                            con=pred_vol) 

market_volatility_pred.drop(columns='index', inplace=True)
market_volatility_pred.set_index(pd.date_range(start='1992-04-01', end='2022-12-31', freq="ME"), inplace=True)



#################################################################
########################### LOAD DATA ###########################
#################################################################

# Load data
factor_returns, _ , transaction_costs, factor_weights, stock_returns, PERMNOs = load_data()

# Exclude I/A and ROE factors
factor_returns.drop(columns=['I/A', 'ROE'], inplace=True)


# Access the database
crsp_compustat = sqlite3.connect(database="/Users/Ilyas/Documents/Mémoire/crsp_compustat.sqlite")

# Import the CRSP monthly data
stock_prices =  pd.read_sql_query(
                            sql="SELECT permno, date, prc from crsp_m",
                            con=crsp_compustat    
                            )

stock_prices.drop_duplicates(subset=['permno','date'],inplace=True)
stock_prices = stock_prices.astype({'permno': int, 'prc':float})

stock_prices = stock_prices.pivot_table(index='date', columns='permno', values='prc')
stock_prices = stock_prices.abs()
stock_prices.dropna(how='all', axis=1)
stock_prices.index = pd.to_datetime(stock_prices.index) + MonthEnd(0)

#################################################################
########################## PREPARE DATA #########################
#################################################################

# Reindex the data so that they all share the same row index

# Get the list of unique dates
unique_dates = pd.date_range(start='1992-04-01', end='2022-12-31', freq="ME")


# Reindex the dataframes so that they all share the same row and column indices
for i in range(len(factor_weights)):
    factor_weights[i] = reindex_weights(factor_weights[i], row=unique_dates, column=PERMNOs, fill=0)

transaction_costs = reindex_weights(transaction_costs, column=PERMNOs, row=unique_dates, fill=0)
stock_returns = reindex_weights(stock_returns, column=PERMNOs, row=unique_dates, fill=0)
factor_returns = reindex_weights(factor_returns, row=unique_dates, fill=0)
market_volatility = reindex_weights(market_volatility_pred, row=unique_dates, fill=0)
stock_prices = reindex_weights(stock_prices, column=PERMNOs, row=unique_dates, fill=0)

market_volatility = market_volatility.shift(-1)

# Fill with 0 for the matrix multiplications
factor_returns.fillna(0.0, inplace=True)
stock_returns.fillna(0.0, inplace=True)
stock_prices.fillna(0.0, inplace=True)

# Transaction costs are half of the spread
transaction_costs = transaction_costs.div(2)

#################################################################
###################### PRECOMPUTE VARIABLES #####################
#################################################################


# Compute r_ext, shape: (T, 2 x K)
extended_factor_returns = precompute_extended_factor_returns(factor_returns, market_volatility)


# Get the sample mean of each factor for the in-sample analysis, shape: (T, 2 x K)
mu_ext = extended_factor_returns.expanding(min_periods=1).mean()


# Desired order of the factors, to facilitate further analysis
desired_order = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM', 'BAB',
                 'MarketVol', 'SMBVol', 'HMLVol', 'RMWVol', 'CMAVol', 'MOMVol','BABVol']

# Get the sample covariance of each factor for the in-sample analysis, shape: (T, 1), and each cell contains a matrix of shape (2 x K, 2 x K)
sigma_ext = expanding_window_covariance(extended_factor_returns, desired_order)
 

# Compute the volatility-managed weights, shape: (2 x K), and each element inside has a shape (T, N)
factor_weights = precompute_weights(factor_weights, market_volatility)



# Compute X_ext for each date, shape: (T, 1), and each cell has a shape (N, 2 x K)
weights_per_dates = create_weights_per_dates(factor_weights)


#################################################################
###################### WEIGHTS OPTIMIZATION #####################
#################################################################




gamma = 5  # Risk-aversion parameter

# Initialisation of the dataframe that will contain the weights of the factors
optimal_eta_df = pd.DataFrame(index=unique_dates, columns=desired_order)

# Initialisation of the dataframe that will contain the portfolio transaction costs
portfolio = pd.DataFrame(index=unique_dates, columns=['Portfolio Value','Transaction Costs'])

# Number of factors
K = len(factor_weights) 

# First guess: equal-weighted, shape: (1, 2 x K)
#wgt_eta = np.array([1.3277229465404619, 0.837389381165301, 1.0460160658429223e-16, 0.0, 4.569037125016756, 0.3472973323564199, 5.4119064118295655e-18	, 1.9278842141570563e-05, 	0.0032872695669896755, 0.017182812730402397, 0.04950084105413666, 1.9972369364017215e-18, 0.005329323296599978,0.0045307431142720634])
wgt_eta = np.ones(K)

# Initialise the cumulative transaction costs
TC = 0

# Initial position, shape: (N, 1)
weights = weights_per_dates.loc[unique_dates[121]] @ wgt_eta.reshape(-1, 1)

# Initialise the number of months of trading, to compute the average transaction costs
t = 1

# Loop over each month of the sample, with an initial window of 10 years
for month in unique_dates[61:]:
    # Keep track of the month processed
    print(f"{month} is being processed.")

    
    # Compute the updated weights (w^+ = X_ext_(t-1) • (1 + r_t)), shape: (N, 1)  
    compounded_weights = weights.reshape(-1, 1) * (stock_returns.loc[month].add(1)).values.reshape(-1, 1) # Shapes: (N, 1) • (N, 1)
    #compounded_weights = weights * (1 + stock_returns.loc[month].values.reshape(-1, 1))
    
    # Compute the optimal weights through the minimisation
    optimal_eta = optimize_eta(mu_ext.loc[month].values.reshape(1, -1), sigma_ext.loc[month].values, transaction_costs.loc[month].values.reshape(-1, 1), compounded_weights, weights_per_dates.loc[month], gamma, TC, t, wgt_eta)
                                                                                                                                                                
    # Keep the weights as an initial guess for the next optimisation, shape: (2 x K, )
    wgt_eta = optimal_eta.x
    
    # Store the weights
    optimal_eta_df.loc[month] = wgt_eta
    
    
    # Update the cumulative transaction costs
    change_weights = (weights_per_dates.loc[month] @ wgt_eta.reshape(-1, 1)) - compounded_weights # Shape: (N, 2 x K) @ (2 x K, 1) - (N, 1)
    TC_eta = np.sum(np.abs(change_weights) * transaction_costs.loc[month].values.reshape(-1, 1)) # Shape: sum(|(N, 1) • (N, 1)|) = (1, 1)
    
    # Do not consider the transaction costs to get the first position
    if t == 1:
        TC = 0
    else: # Add the transaction costs associated with the rebalancing to the cumulative transaction costs
        TC += TC_eta.item()
    

    # Store the transaction costs
    portfolio.loc[month,'Transaction Costs'] = TC_eta.item()
    
    # Update the out-of-sample portfolio values
    portfolio_value = np.sum((weights_per_dates.loc[month] @ wgt_eta.reshape(-1, 1)) * stock_prices.loc[month].values.reshape(-1,1))
    print(f'Portfolio Value: {portfolio_value}')
    portfolio.loc[month,'Portfolio Value'] = portfolio_value
    
    # Compute X_ext @ eta, shape: (N, 1)
    weights = weights_per_dates.loc[month] @ wgt_eta.reshape(-1, 1) # Shapes: (N, 2 x K) @ (2 x K, 1)
    
    # Update the month counter
    t += 1
    




#################################################################
###################### DISPLAY THE RESULTS ######################
#################################################################

# Compute the portfolio returns
ret = optimal_eta_df.shift(1).dropna(axis=0).mul(extended_factor_returns)

# Get the returns net of transaction costs
port_ret = ret.dropna().sum(axis=1).sub(portfolio["Transaction Costs"])


#cumret = port_ret.dropna(axis=0).add(1).cumprod().div(10000)
conn = sqlite3.connect("/Users/Ilyas/Documents/Mémoire/market.sqlite")
mret =  pd.read_sql_query(
                            sql="SELECT * from mret",
                            con=conn,    
                            index_col='date')


mret.index = pd.to_datetime(mret.index)

# Adjust the returns series so that it has the same volatility as the market
adjustment_factor = mret.std()/port_ret.std()
port_ret *= adjustment_factor.values.item()


# Load the regular volatility-managed portfolio returns
# port_ret_CMV = pd.read_sql_query(
#                             sql="SELECT * from CMVreturns",
#                             con=conn2,    
#                             index_col='date') 
# port_ret_CMV.index = pd.to_datetime(port_ret_CMV.index)
port_ret_CMV = cmv_ret
port_ret_CMV *= adjustment_factor.values.item()


# Create the cumulative returns series
cumret = port_ret.dropna(axis=0).add(1).cumprod().sub(1)
mcumret = mret.loc[cumret.index[0]:].dropna(axis=0).add(1).cumprod().sub(1)
cmvcumret = port_ret_CMV.loc[cumret.index[0]:].dropna(axis=0).add(1).cumprod().sub(1)

# Let the series start at the same point
cumret += 1 - cumret.iloc[0] # The first return will be 0 on the log-axis
mcumret += 1 - mcumret.iloc[0]
cmvcumret += 1 - cmvcumret.iloc[0]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(cumret, c='dodgerblue', label='CMV - with prediction')  # Replace with your actual label
plt.plot(mcumret, c='lightgrey', label='Market' )
plt.plot(cmvcumret, c='crimson', label='CMV - regular')

# Set y-axis to logarithmic scale
plt.yscale('log')

# Adding labels and title
plt.xlabel('Year')
plt.ylabel('Cumulative Returns (log scale)')
plt.title('Cumulative Returns of Multifactor Portfolios')
plt.legend()

#plt.savefig("CMVvsMarket.png", bbox_inches="tight")
# Display the plot
plt.show()

# Display the Sharpe ratios, assuming a risk-free rate equal to 0.047
print(f'The Sharpe ratio of the volatility-managed portfolio with prediction is {(port_ret.mean())/port_ret.std()}')
print(f'The Sharpe ratio of the regular volatility-managed portfolio is {(cmv_ret.iloc[:-1].mean())/cmv_ret.iloc[:-1].std()}')
print(f'The Sharpe ratio of the market portfolio is {(mret.mean())/mret.std()}')



    
#################################################################
####################### COMPARISON MODELS #######################
#################################################################


def SR_pval(model1ret, model2ret, n):
    rs = RandomState(1234)
    bs = StationaryBootstrap(5, model1ret, model2ret, random_state=rs)
    
    # Array to store the differences in Sharpe ratios
    diff_distr = np.zeros(n)
    
    for data in bs.bootstrap(n):
        bs_model1ret = data[0][0]
        bs_model2ret = data[0][1]
        
        # Computing the Sharpe ratio of each series
        sr1 = bs_model1ret.mean() / bs_model1ret.std()
        sr2 = bs_model2ret.mean() / bs_model2ret.std()
        
        # Store the difference in Sharpe ratios
        diff_distr[i] = sr1 -  sr2
        
    # Calculate the p-value
    negative_count = (diff_distr < 0).sum()
    p_value = negative_count / n
    
    return p_value, diff_distr

p_val, diff_distr = SR_pval(port_ret.dropna(), cmv_ret.dropna().iloc[:-1], 10000)
print(f'The p-value is {p_val}')