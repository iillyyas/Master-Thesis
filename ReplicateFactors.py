import itertools
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import sqlite3


#################################################################
########################### CRSP DATA ###########################
#################################################################

# Access the database
crsp_compustat = sqlite3.connect(database="/Users/Ilyas/Documents/Mémoire/crsp_compustat.sqlite")

# Import the CRSP monthly data
crsp =  pd.read_sql_query(
                            sql="SELECT * FROM crsp_m",
                            con=crsp_compustat    
                            )

# Rename columns 
ccols = {'permno': 'PERMNO',
         'permco' : 'PERMCO',
         'ret' : 'RET',
         'prc' : 'PRC',
         'shrout' : 'SHROUT',
         'exchcd' : 'EXCHCD',
         'shrcd' : 'SHRCD'}
crsp = crsp.rename(columns = ccols)

# Dictionary to type 32bit
dictotype32 = {'PERMNO': np.int32,
               'PERMCO': np.int32,
               'EXCHCD': np.int32,
               'SHRCD': np.int32,
               'RET': np.float32}

# Convert the variables to the required type
crsp = crsp.astype(dictotype32)

# Convert the date to a datetime format
crsp['date'] = pd.to_datetime(crsp['date'])

# Line up date to be end of month
crsp['date'] = crsp['date'] + MonthEnd(0)

# Ensure that price data is positive
crsp['PRC'] = np.abs(crsp['PRC'])

# Compute the market cap at the PERMNO level, and set it to NaN if it is 0
crsp['CAP'] = crsp['PRC'] * crsp['SHROUT']
crsp['CAP'] = np.where(crsp['CAP']==0, np.nan, crsp['CAP'])

# Drop duplicates
crsp.drop_duplicates(subset=['date', 'PERMNO'], ignore_index=True, inplace=True)
crsp = crsp.sort_values(by=['PERMNO', 'date']).reset_index(drop=True)

# Define the month and year column
crsp['month'] = crsp['date'].dt.month
crsp['year'] = crsp['date'].dt.year

# Returns that are -66, -77, -88, -99 are mapped to null
crsp['RET'] = np.where(crsp['RET'] < -1, np.nan, crsp['RET'])

# Isolate the December and June market cap
crsp['DEC_CAP'] = crsp.loc[crsp['month'] == 12, 'CAP']
crsp['JUN_CAP'] = crsp.loc[crsp['month'] == 6, 'CAP']

# Fill the missing values with the preceding year's market cap
crsp['DEC_CAP'] = crsp.groupby('PERMNO')['DEC_CAP'].ffill(limit=11)
crsp['JUN_CAP'] = crsp.groupby('PERMNO')['JUN_CAP'].ffill(limit=11)


#################################################################
##################### COMPUSTAT ANNUAL DATA #####################
#################################################################

# Import the Compustat data
comp = pd.read_sql_query(
                            sql="SELECT * FROM comp",
                            con=crsp_compustat    
                            )

# Dictionary to type 32bit
comptotype32 = {'cusip': str,
                'gvkey' : np.int32,
                'at' : np.float32,         
                'ceq' : np.float32,       
                'cogs' : np.float32,         
                'lt' : np.float32,         
                'mib' : np.float32,         
                'pstk' : np.float32,        
                'pstkl' : np.float32, 
                'pstkrv' : np.float32,   
                'sale' : np.float32,       
                'seq' : np.float32,         
                'txdb' : np.float32,       
                'txditc' : np.float32,      
                'xint' : np.float32,        
                'xsga' : np.float32}     
    
# Convert the variables to the required type
comp = comp.astype(comptotype32)

# Rename the date column and convert it to an datetime format
comp.rename(columns={'datadate':'date_compa'},inplace=True)
comp['date_compa'] = pd.to_datetime(comp['date_compa'])

# Line up date to be end of month
comp['date_compa'] = comp['date_compa'] + MonthEnd(0)

# Drop duplicates and sort by gvkey and date_jun
comp = comp.drop_duplicates(subset=['gvkey', 'date_compa'])
comp = comp.sort_values(by=['gvkey', 'date_compa'])

# Counting the number of years a company has been in Compustat
comp['count'] = comp.groupby(['gvkey']).cumcount()

## Compute Operating Profitability
# Compute the book value of equity (BE)
# Calculate SBE (SEQ, or CEQ + PSTK, or CEQ + AT - LT - MIB, or NaN)
comp['SBE'] = np.where(comp['seq'].notna(), comp['seq'], 
                       np.where(comp['ceq'].notna(),
                                np.where(comp['pstk'].notna(), comp['ceq'] + comp['pstk'], comp['ceq'] + comp['at'] - comp['lt'] - comp['mib']),
                                np.nan))

# Calculate BVPS (PSTKRV, or PSTKL, or PSTK, or NaN)
comp['BVPS'] = np.where(comp['pstkrv'].notna(), comp['pstkrv'], 
                        np.where(comp['pstkl'].notna(), comp['pstkl'], 
                                 np.where(comp['pstk'].notna(), comp['pstk'], np.nan)))

# Calculate DT (TXDITC, or TXDB + ITCB)
comp['DT'] = np.where(comp['txditc'].notna(), comp['txditc'], comp['txdb'] + comp['itcb'])

# BE
comp['BE'] = comp['SBE'] - comp['BVPS'].fillna(0) + comp['DT']

# Ensure positive book value of equity
comp['BE'] = np.where(comp['BE'] < 0, np.nan, comp['BE'])

# Compute Operating Profits
# Check if at least one of cogs, xsga, or xint exists
one_exists = comp[['cogs', 'xsga', 'xint']].notna().any(axis=1)

# operpro
comp['operpro'] = np.where(comp['sale'].notna() & one_exists, 
                           comp['sale'] - comp['cogs'].fillna(0) - comp['xsga'].fillna(0) - comp['xint'].fillna(0), 
                           np.nan)

# Operating Profitability
comp['OP'] = np.where(comp['BE'] > 0, comp['operpro'] / comp['BE'], np.nan)

## Compute Investment
# Calculate the lagged total assets by one year
comp['at_lag1'] = comp.groupby('gvkey')['at'].shift(1)

# Compute the investment variable
comp['INV'] = np.where((comp['at'] > 0) & (comp['at_lag1'] > 0), 
                       comp['at'] / comp['at_lag1'] - 1, 
                       np.nan)

# Rename GVKEY
comp.rename(columns={'gvkey':'GVKEY'}, inplace=True)


#################################################################
################### COMPUSTAT QUARTERLY DATA ####################
#################################################################

# Import the Compustat Quarterly data
compq = pd.read_sql_query(
                            sql="SELECT * FROM compq",
                            con=crsp_compustat    
                            )

# Dictionary to type 32bit
compqtotype32 = {'cusip' : str,        
                'atq' : np.float32,         
                'ceqq' : np.float32,               
                'ltq' : np.float32,                 
                'pstkq' : np.float32,              
                'seqq' : np.float32,             
                'txditcq' : np.float32,      
                'ibq' : np.float32}

# Convert the variables to the required type  
compq = compq.astype(compqtotype32)

# Rename the date column and convert it to an datetime format
compq.rename(columns={'datadate':'date_compq'},inplace=True)
compq['date_compq'] = pd.to_datetime(compq['date_compq'])

# Line up date to be end of month
compq['date_compq'] = compq['date_compq'] + MonthEnd(0)

# Drop duplicates and sort by cusip and datadate
compq = compq.drop_duplicates(subset=['cusip', 'date_compq'])
compq = compq.sort_values(by=['cusip', 'date_compq'])

## Compute the book value of equity (BEQ)
# Calculate SBES (SEQQ, or CEQQ + PSTKQ, or ATQ - LTQ, or NaN)
condition1 = compq['ceqq'].notna() & compq['pstkq'].notna()
result1 = np.where(condition1, compq['ceqq'] + compq['pstkq'], compq['atq'] - compq['ltq'])

compq['SBEQ'] = np.where(compq['seqq'].notna(), compq['seqq'], 
                         np.where(compq['ceqq'].notna(), result1, np.nan))

# Calculate BVPSS (PSTKRQ, or PSTKQ, or NaN)
compq['BVPSQ'] = np.where(compq['pstkrq'].notna(), compq['pstkrq'], 
                         np.where(compq['pstkq'].notna(), compq['pstkq'], np.nan))

# Calculate DTS
compq['DTQ'] = compq['txditcq']

# Compute BEQ
compq['BEQ'] = compq['SBEQ'] - compq['BVPSQ'].fillna(0) + compq['DTQ'].fillna(0)

# Keep only positive book value of equity
#compq['BEQ'] = np.where(compq['BEQ'] < 0, np.nan, compq['BEQ'])

# Lag the Book Value of Equity by one quarter
compq['BEQ_lag1'] = compq.groupby('cusip')['BEQ'].shift(1)

# Compute the ROE
compq['ROE'] = compq['ibq'] / compq['BEQ_lag1']


#################################################################
################# MERGE COMPUSTAT AND CRSP DATA #################
#################################################################

# Import the link table
link_table = pd.read_sql_query(
                                sql="SELECT * FROM ccm",
                                con=crsp_compustat    
                                )

# Rename the columns
link_table.rename(columns={'gvkey':'GVKEY', 'permno':'PERMNO', 'lpermco': 'PERMCO', 'linkdt':'LINKDT', 'linkenddt':'LINKENDDT'}, inplace=True)

# Convert the GVKEY to a consistent type and LINKDT and LINKENDDT to a datetime format
link_table['GVKEY'] = link_table['GVKEY'].astype(np.int32)
link_table['LINKDT'] = pd.to_datetime(link_table['LINKDT'])
link_table['LINKENDDT'] = pd.to_datetime(link_table['LINKENDDT'])
link_table['LINKENDDT'] = link_table['LINKENDDT'].fillna(value = crsp.date.max())

# Inner merge between the link_table and the CRSP data where dates are in the bounds LINKDT and LINKENDDT
crsp_ccm = (crsp
            .merge(link_table, how="left", on=['PERMNO', 'PERMCO'])
            .query("(date >= LINKDT) & (date <= LINKENDDT)")
            )

# Left merge crsp_ccm with Annual Compustat
crsp_comp_ccm = pd.merge(crsp_ccm, comp, how='left', left_on=['GVKEY','date'], right_on=['GVKEY', 'date_compa'])


# Left merge crsp_comp_ccm  with Quarterly Compustat
crsp_compq = pd.merge(crsp_comp_ccm, compq, how='left', left_on=['cusip','date'], right_on=['cusip', 'date_compq'])

# Deleting unused variables
del ccols, dictotype32, crsp, comp, comptotype32, one_exists, compq, compqtotype32
del condition1, result1, link_table, crsp_ccm, crsp_comp_ccm


#################################################################
############# SORT PORTFOLIOS INTO CHARACTERISTICS ##############
#################################################################


# Keep only ordinary shares
crsp_compq = crsp_compq[crsp_compq['SHRCD'].isin(set([10,11]))].copy()

# Create a column that contains 1 if the stock trades on the NYSE
crsp_compq['NYSE'] = np.where(crsp_compq['EXCHCD'] == 1, 1, 0)

# Get the market equity aggregated at the PERMCO level
crsp_compq['ME'] = crsp_compq.groupby(['GVKEY', 'date'])['JUN_CAP'].transform('sum')
crsp_compq['ME_dec'] = crsp_compq.groupby(['GVKEY', 'date'])['DEC_CAP'].transform('sum')

# Book to Market
crsp_compq['BtM'] = crsp_compq['BE'] / crsp_compq['ME_dec']

# Fill the missing values up to 1 year ahead so that they are available for the bucket attribution
crsp_compq['BtM'] = crsp_compq.groupby('PERMNO')['BtM'].ffill(limit=11)
crsp_compq['OP'] = crsp_compq.groupby('PERMNO')['OP'].ffill(limit=11)
crsp_compq['INV'] = crsp_compq.groupby('PERMNO')['INV'].ffill(limit=11)


# Filter the data for June and NYSE equals 1
june_nyse_data = crsp_compq[(crsp_compq['date'].dt.month == 6) & (crsp_compq['NYSE'] == 1)]

## Size characteristic
# Group by date and compute the median of 'ME'
median_me_by_date = june_nyse_data.groupby('date')['ME'].median().reset_index()
median_me_by_date.rename(columns={'ME': 'Size breakp'}, inplace=True)

# Merge the median values back to the original DataFrame
crsp_compq = crsp_compq.merge(median_me_by_date, on='date', how='left')

# Create a column that contains Small if the stock has a small cap and Big if it has a big cap
crsp_compq['Size_bucket'] = np.where(crsp_compq['ME'] <= crsp_compq['Size breakp'], 'Small',
                                     np.where(crsp_compq['ME'] > crsp_compq['Size breakp'], 'Big', 0)) 

# Ensure that NaNs are correctly recognized as np.nan in the 'Size_bucket' column
crsp_compq['Size_bucket'] = crsp_compq['Size_bucket'].replace('0', np.nan)


## Book to Market characteristic
# Get the breakpoints for the Book to Market
quantiles_BtM_by_date = june_nyse_data.groupby('date')['BtM'].quantile(q=[.3, .7]).unstack().reset_index()
quantiles_BtM_by_date.rename(columns={0.3: 'BtM breakp 0.3', 0.7: 'BtM breakp 0.7'}, inplace=True)

# Merge the quantile values back to the original DataFrame
crsp_compq = crsp_compq.merge(quantiles_BtM_by_date, on='date', how='left')

# Create a column that contains Low if the stock has a small BtM and High if it has a big BtM
crsp_compq['BtM_bucket'] = np.where(crsp_compq['BtM'] <= crsp_compq['BtM breakp 0.3'], 'Low', 
                                    np.where(crsp_compq['BtM'] > crsp_compq['BtM breakp 0.7'], 'High', 
                                             np.where((crsp_compq['BtM'] > crsp_compq['BtM breakp 0.3']) & (crsp_compq['BtM'] <= crsp_compq['BtM breakp 0.7']), 'Mid BtM', 0) )) 



## Operating Profitability characteristic
# Get the breakpoints for the Operating Profitability
quantiles_OP_by_date = june_nyse_data.groupby('date')['OP'].quantile(q=[.3, .7]).unstack().reset_index()
quantiles_OP_by_date.rename(columns={0.3: 'OP breakp 0.3', 0.7: 'OP breakp 0.7'}, inplace=True)

# Merge the quantile values back to the original DataFrame
crsp_compq = crsp_compq.merge(quantiles_OP_by_date, on='date', how='left')

# Create a column that contains Weak if the stock has a small OP and Robust if it has a big OP
crsp_compq['OP_bucket'] = np.where(crsp_compq['OP'] <= crsp_compq['OP breakp 0.3'], 'Weak', 
                                    np.where(crsp_compq['OP'] > crsp_compq['OP breakp 0.7'], 'Robust', 
                                             np.where((crsp_compq['OP'] > crsp_compq['OP breakp 0.3']) & (crsp_compq['OP'] <= crsp_compq['OP breakp 0.7']), 'Mid OP', 0) )) 

## Investment Characteristic
# Get the breakpoints for the Investment
quantiles_INV_by_date = june_nyse_data.groupby('date')['INV'].quantile(q=[.3, .7]).unstack().reset_index()
quantiles_INV_by_date.rename(columns={0.3: 'INV breakp 0.3', 0.7: 'INV breakp 0.7'}, inplace=True)

# Merge the quantile values back to the original DataFrame
crsp_compq = crsp_compq.merge(quantiles_INV_by_date, on='date', how='left')

# Create a column that contains Conservative if the stock has a small INV and Aggressive if it has a big INV
crsp_compq['INV_bucket'] = np.where(crsp_compq['INV'] <= crsp_compq['INV breakp 0.3'], 'Conservative', 
                                    np.where(crsp_compq['INV'] > crsp_compq['INV breakp 0.7'], 'Aggressive', 
                                             np.where((crsp_compq['INV'] > crsp_compq['INV breakp 0.3']) & (crsp_compq['INV'] <= crsp_compq['INV breakp 0.7']), 'Mid INV', 0) )) 

crsp_compq['INV_bucket'] = crsp_compq['INV_bucket'].replace('0', np.nan)

# Deleting unused variables
del june_nyse_data, median_me_by_date, quantiles_BtM_by_date, quantiles_OP_by_date, quantiles_INV_by_date


#################################################################
####################### COMPUTE FF FACTORS ######################
#################################################################

def compute_cap_weighted_returns(crsp_compq, characteristics, columns):
    """
    Compute monthly cap-weighted returns for portfolios rebalanced annually formed based on given characteristics and 
    output an additional DataFrame with weights for the factors.

    Parameters
    ----------
    crsp_compq : Pandas DataFrame
        Contains the data to compute the returns (date, PERMNO, RET, ME).
    characteristics : List of tuples
        Characteristics on which to do the portfolio sorts.
    columns : List of strings
        Contains in which are stored the buckets of stocks.

    Returns
    -------
    pivoted_returns : Pandas DataFrame
        Returns of portfolios sorted independently on characteristics.
    weights_dfs : Dictionary of Pandas DataFrame
        Weights of the sorted portfolios

    """
    
    # Initialize an empty DataFrame to store the cap-weighted returns
    cap_weighted_returns = pd.DataFrame(columns=['date'] + columns + ['cap_weighted_return'])
    
    # Initialize a dictionary to store weights for each characteristic
    weights_data = {char: [] for char in characteristics}
    
    # Extract unique years from the 'date' column
    years = crsp_compq['date'].dt.year.unique()

    # Loop over each year
    for year in years:
        # Loop over each set of characteristic values
        for characteristic_values in characteristics:
            # Filter stocks based on the date and characteristic values
            filter_condition = (crsp_compq['date'] == pd.Timestamp(year, 6, 30))
            for col, val in zip(columns, characteristic_values):
                filter_condition &= (crsp_compq[col] == val)
            
            # Select stocks that meet the filter condition
            june_stocks = crsp_compq.loc[filter_condition]
            
            if not june_stocks.empty:
                # Get unique stock identifiers (PERMNO)
                stock_ids = june_stocks['PERMNO'].unique()
                
                # Define the start and end date for the next 12 months
                start_date = pd.Timestamp(year, 7, 1)
                end_date = pd.Timestamp(year + 1, 6, 30)
                
                # Filter stocks that fall within the 12-month period and are in the selected stock IDs
                period_stocks = crsp_compq.loc[
                    (crsp_compq['date'] >= start_date) &
                    (crsp_compq['date'] <= end_date) &
                    (crsp_compq['PERMNO'].isin(stock_ids))
                ]
                
                # Merge with June stocks to carry forward June characteristics
                period_stocks = period_stocks.merge(june_stocks[['PERMNO']], on='PERMNO', how='left')
                
                # Compute the total market capitalization (cap) of the period stocks
                total_cap = period_stocks['JUN_CAP'].sum()
                
                # Calculate the weight for each stock based on its June capitalization
                period_stocks = period_stocks.assign(weight=period_stocks['JUN_CAP'] / total_cap)
                
                # Append the weights data to the corresponding characteristic value
                weights_data[characteristic_values].append(period_stocks[['date', 'PERMNO', 'weight']])
                
                # Compute the weighted return for each stock
                period_stocks['weighted_return'] = period_stocks['RET'] * period_stocks['weight']
                
                # Group by date and sum the weighted returns to get the cap-weighted return for each date
                grouped = period_stocks.groupby('date')['weighted_return'].sum().reset_index()
                
                # Rename columns and add characteristic values to the grouped DataFrame
                grouped.columns = ['date', 'cap_weighted_return']
                for col, val in zip(columns, characteristic_values):
                    grouped[col] = val
                
                # Append the grouped DataFrame to the cap_weighted_returns DataFrame
                cap_weighted_returns = pd.concat([cap_weighted_returns, grouped], ignore_index=True)
    
    # Sort the cap_weighted_returns DataFrame by date
    cap_weighted_returns.sort_values(by='date', inplace=True)
    cap_weighted_returns.reset_index(drop=True, inplace=True)

    # Pivot the cap_weighted_returns DataFrame to have dates as rows and characteristic combinations as columns
    pivoted_returns = cap_weighted_returns.pivot_table(
        index='date',
        columns=columns,
        values='cap_weighted_return'
    )

    # Flatten the multi-index columns
    pivoted_returns.columns = ['_'.join(map(str, col)).replace(' ', '').replace('_', '') for col in pivoted_returns.columns.values]
    pivoted_returns.reset_index(inplace=True)

    # Create a dictionary of DataFrames containing weights for each characteristic combination
    weights_dfs = {}
    for characteristic_values, weight_list in weights_data.items():
        if weight_list:
            weights_df = pd.concat(weight_list)
            weights_df = weights_df.pivot_table(index='date', columns='PERMNO', values='weight')
            weights_dfs[characteristic_values] = weights_df
    
    return pivoted_returns, weights_dfs


# Import the list of PERMNOs
PERMNOs = pd.read_sql_query("SELECT * FROM PERMNOs", crsp_compustat)
PERMNOs = PERMNOs.astype(int)
PERMNOs = PERMNOs.iloc[:,0].to_list()

def reindex_weights(weights_dict, stock_ids):
    """
    Reindex the dataframes in the weights dictionary to have the same columns.

    Parameters
    ----------
    weights_dict: Dictionary 
        Dictionary containing dataframes with portfolio weights.
    stock_ids: List
        List of stock IDs to use for reindexing columns.

    Returns
    ----------
    dict: Dictionary of pandas DataFrames
        The reindexed weights dictionary.
    """
    
    reindexed_weights_dict = {}
    for key in weights_dict.keys():
        reindexed_weights_dict[key] = weights_dict[key].reindex(columns=PERMNOs).fillna(0)
    return reindexed_weights_dict

# Size factor
size_buckets = [('Small',), ('Big',)]
size_column = ['Size_bucket']
size_factor, weightsSMBdict = compute_cap_weighted_returns(crsp_compq, size_buckets, size_column)
size_factor['SMB'] = size_factor['Small'] - size_factor['Big']

# Compute the SMB weights
weightsSMBdict = reindex_weights(weightsSMBdict, PERMNOs)
weightsSMB = weightsSMBdict[('Small',)].fillna(0) - weightsSMBdict[('Big',)].fillna(0)


# HML factor
size_btm_buckets = [('Small', 'Low'), ('Small', 'High'), ('Big', 'Low'), ('Big', 'High')]
size_btm_columns = ['Size_bucket', 'BtM_bucket']
HML_factor, weightsHMLdict = compute_cap_weighted_returns(crsp_compq, size_btm_buckets, size_btm_columns)
HML_factor['HML'] = 0.5 * (HML_factor['SmallHigh'] + HML_factor['BigHigh']) - 0.5 * (HML_factor['SmallLow'] + HML_factor['BigLow'])

# Compute the HML weights
weightsHMLdict = reindex_weights(weightsHMLdict, PERMNOs)
weightsHML = 0.5 * (weightsHMLdict[('Small', 'High')].fillna(0) + weightsHMLdict[('Big', 'High')].fillna(0)) - 0.5 * (weightsHMLdict[('Small', 'Low')].fillna(0) + weightsHMLdict[('Big', 'Low')].fillna(0))


# RMW factor
size_op_buckets = [('Small', 'Robust'), ('Small', 'Weak'), ('Big', 'Robust'), ('Big', 'Weak')]
size_op_columns = ['Size_bucket', 'OP_bucket']
RMW_factor, weightsRMWdict = compute_cap_weighted_returns(crsp_compq, size_op_buckets, size_op_columns)
RMW_factor['RMW'] = 0.5 * (RMW_factor['SmallRobust'] + RMW_factor['BigRobust']) - 0.5 * (RMW_factor['SmallWeak'] + RMW_factor['BigWeak'])

# Compute the RMW weights
weightsRMWdict = reindex_weights(weightsRMWdict, PERMNOs)
weightsRMW = 0.5 * (weightsRMWdict[('Small', 'Robust')].fillna(0) + weightsRMWdict[('Big', 'Robust')].fillna(0)) - 0.5 * (weightsRMWdict[('Small', 'Weak')].fillna(0) + weightsRMWdict[('Big', 'Weak')].fillna(0))


# CMA Factor
size_inv_buckets = [('Small', 'Conservative'), ('Small', 'Aggressive'), ('Big', 'Conservative'), ('Big', 'Aggressive')]
size_inv_columns = ['Size_bucket', 'INV_bucket']
CMA_factor, weightsCMAdict = compute_cap_weighted_returns(crsp_compq, size_inv_buckets, size_inv_columns)
CMA_factor['CMA'] = 0.5 * (CMA_factor['SmallConservative'] + CMA_factor['BigConservative']) - 0.5 * (CMA_factor['SmallAggressive'] + CMA_factor['BigAggressive'])

# Compute the CMA weights
weightsCMAdict = reindex_weights(weightsCMAdict, PERMNOs)
weightsCMA = 0.5 * (weightsCMAdict[('Small', 'Conservative')].fillna(0) + weightsCMAdict[('Big', 'Conservative')].fillna(0)) - 0.5 * (weightsCMAdict[('Small', 'Aggressive')].fillna(0) + weightsCMAdict[('Big', 'Aggressive')].fillna(0))


# Set the date as the index
size_factor.set_index('date', inplace=True)
HML_factor.set_index('date', inplace=True)
RMW_factor.set_index('date', inplace=True)
CMA_factor.set_index('date', inplace=True)

# Deleting unused variables
del size_buckets, size_column, weightsSMBdict, size_btm_buckets, size_btm_columns, weightsHMLdict
del size_op_buckets, size_op_columns, weightsRMWdict, size_inv_buckets, size_inv_columns, weightsCMAdict




#################################################################
####################### COMPUTE FF FACTORS ######################
#################################################################

# Load the Fama French Factors
ff_original_fact = pd.read_csv("/Users/Ilyas/Documents/Mémoire/_FF 5 Factors_2x3.csv", skiprows=3, nrows=729)
ff_original_fact.rename(columns={"Unnamed: 0":"date"}, inplace="True")
ff_original_fact["date"] = pd.to_datetime(ff_original_fact["date"], format="%Y%m") + MonthEnd(0)
ff_original_fact.set_index(ff_original_fact["date"], drop=True, inplace=True)

# Merge the Risk-Free rate with the stock data
crsp_compq = crsp_compq.merge(ff_original_fact.RF, how='left', on='date')

# Shift the market cap to get the weights from the previous month
crsp_compq['ME_shifted'] = crsp_compq.groupby('PERMNO')['ME'].shift(1)

ME_shifted_total = crsp_compq.groupby('date')['ME_shifted'].apply(lambda x: x.sum())
ME_shifted_total.name = 'ME_shifted_total'
crsp_compq = crsp_compq.merge(ME_shifted_total, how='left', left_on='date', right_index=True)

# Calculate weights
crsp_compq['weight'] = crsp_compq['ME_shifted'].div(crsp_compq['ME_shifted_total'])

# Calculate portfolio returns
crsp_compq['Mkt-RF'] = (crsp_compq['RET'] - crsp_compq['RF'].div(100)) * crsp_compq['weight']

# Compute the portfolio return for each month
market_factor = crsp_compq.groupby('date')['Mkt-RF'].sum()


# Create a DataFrame to store weights for each month
weightsMarket = crsp_compq.pivot_table(index='date', columns='PERMNO', values='weight').fillna(0)


#################################################################
######### COMPARE THE RESULTS WITH THE ORIGINAL FACTORS #########
#################################################################

# Merge the replicated factors with the original ones into a single dataframe
ffcomp = pd.merge(ff_original_fact, market_factor, how='inner', left_index=True, right_index=True)
ffcomp = pd.merge(ffcomp, size_factor['SMB'], how='inner', left_index=True, right_index=True)
ffcomp = pd.merge(ffcomp, HML_factor['HML'], how='inner', left_index=True, right_index=True)
ffcomp = pd.merge(ffcomp, RMW_factor['RMW'], how='inner', left_index=True, right_index=True)
ffcomp = pd.merge(ffcomp, CMA_factor['CMA'], how='inner', left_index=True, right_index=True)

# Get the correlation between the original and the replicated factor (after 1966)
ffcomp66 = ffcomp[ffcomp['date']>='01/01/1966']

print('Correlation with the original series')
print('Market', f"{np.corrcoef(ffcomp66['Mkt-RF_x'], ffcomp66['Mkt-RF_y'])[1][0]:.0%}")
print('SMB', f"{np.corrcoef(ffcomp66['SMB_x'], ffcomp66['SMB_y'])[1][0]:.0%}")
print('HML', f"{np.corrcoef(ffcomp66['HML_x'], ffcomp66['HML_y'])[1][0]:.0%}")
print('RMW', f"{np.corrcoef(ffcomp66['RMW_x'], ffcomp66['RMW_y'])[1][0]:.0%}")
print('CMA', f"{np.corrcoef(ffcomp66['CMA_x'], ffcomp66['CMA_y'])[1][0]:.0%}")


# Removing unused variables
crsp_compq.drop(['month', 'year', 'linktype', 'linkprim', 'LINKDT', 'LINKENDDT', 'date_compa', 'at',
'pstkl', 'txditc', 'pstkrv', 'seq', 'pstk', 'ceq', 'lt', 'txdb', 'itcb',
'cusip', 'capx', 'oancf', 'sale', 'cogs', 'xint', 'xsga', 'dp', 'mib',
'fyr', 'fyear', 'SBE', 'BVPS', 'DT','operpro','seqq', 'txditcq', 'ceqq', 'pstkq', 'atq',
'ltq', 'pstkrq', 'SBEQ', 'BVPSQ', 'DTQ'], inplace=True, axis=1)

del ffcomp, ffcomp66

#################################################################
##################### REPLICATE HXZ FACTORS #####################
#################################################################

# Removing financial firms and firms with negative book equity
crsp_compq = crsp_compq[~crsp_compq['siccd'].between(6000, 6999)]

crsp_compq = crsp_compq[~crsp_compq['BEQ'].lt(0)]

# Ensure 'rdq' is in datetime format
crsp_compq.loc[:, 'rdq'] = np.where((pd.isnull(crsp_compq['rdq'])) & (~pd.isnull(crsp_compq['ibq'])), crsp_compq['date'], crsp_compq['rdq'])
crsp_compq['rdq'] = pd.to_datetime(crsp_compq['rdq'])


# Calculate the target date for ROE
crsp_compq['target_date'] = crsp_compq['rdq'] + MonthEnd(2)

# Filter for NYSE stocks
crsp_compq_nyse = crsp_compq[crsp_compq['NYSE'] == 1]

# Prepare a DataFrame that shifts the ROE values based on the target_date
crsp_compq_shifted = crsp_compq[['PERMNO', 'ROE', 'target_date']].copy()
crsp_compq_shifted.rename(columns={'ROE': 'shifted_ROE', 'target_date': 'date'}, inplace=True)


# Merge shifted ROE back to the main DataFrame
crsp_compq = crsp_compq.merge(crsp_compq_shifted, on=['PERMNO', 'date'], how='left')

# Fill the ROE column until the last date where it can be used
crsp_compq['ROE'] = crsp_compq.groupby('PERMNO')['ROE'].ffill(limit=5)
crsp_compq['shifted_ROE'] = crsp_compq.groupby('PERMNO')['shifted_ROE'].ffill(limit=5)

# Filter the data to ensure the ROE value is valid within the 6-month window
crsp_compq['ROE_valid'] = np.where(crsp_compq['ROE'] == np.nan, -np.inf, crsp_compq['shifted_ROE'])
crsp_compq['ROE_valid'] = crsp_compq['ROE_valid'].replace(-np.inf, np.nan)


# Filter the data for June and NYSE equals 1
nyse_data = crsp_compq[(crsp_compq['NYSE'] == 1)]

## Return On Equity characteristic
# Get the breakpoints for the ROE
quantiles_ROE_by_date = nyse_data.groupby('date')['ROE_valid'].quantile(q=[.3, .7]).unstack().reset_index()
quantiles_ROE_by_date.rename(columns={0.3: 'ROE breakp 0.3', 0.7: 'ROE breakp 0.7'}, inplace=True)

# Merge the quantile values back to the original DataFrame
crsp_compq = crsp_compq.merge(quantiles_ROE_by_date, on='date', how='left')

# Create a column that contains LowROE if the stock has a small ROE and HighROE if it has a big ROE
crsp_compq['ROE_bucket'] = np.where(crsp_compq['ROE_valid'] <= crsp_compq['ROE breakp 0.3'], 'LowROE', 
                                    np.where(crsp_compq['ROE_valid'] > crsp_compq['ROE breakp 0.7'], 'HighROE', 
                                             np.where((crsp_compq['ROE_valid'] > crsp_compq['ROE breakp 0.3']) & (crsp_compq['ROE_valid'] <= crsp_compq['ROE breakp 0.7']), 'Mid ROE', 0) )) 


# Forward fill the yearly buckets for Size and INV
crsp_compq['Size_bucket'] = crsp_compq.groupby('PERMNO')['Size_bucket'].ffill(limit=11)
crsp_compq['INV_bucket'] = crsp_compq.groupby('PERMNO')['INV_bucket'].ffill(limit=11)



def compute_cap_weighted_returns_monthly_rebalance(crsp_compq, characteristics, columns):
    """
    Compute monthly cap-weighted returns for portfolios rebalanced monthly formed based on given characteristics and 
    output an additional DataFrame with weights for the factors.

    Parameters:
    crsp_compq: Pandas DataFrame 
        DataFrame containing stock data with necessary columns.
    characteristics (list of tuples): List of tuples where each tuple contains characteristic values.
    columns (list of str): List of columns corresponding to the characteristic values.

    Returns: 
    pivoted_returns: Pandas DataFrame
        DataFrame with cap-weighted returns for each combination of characteristics.
    weights_dfs: Dictionary of DataFrames
        Weights for each characteristic combination.
    """
    # Initialize the list to store results
    results = []
    weights_data = {char: [] for char in characteristics}

    # Extract unique periods (months)
    crsp_compq['period'] = crsp_compq['date'].dt.to_period('M')

    # Compute the cap-weighted returns for each combination of characteristics
    for characteristic_values in characteristics:
        # Create filter conditions for each combination of characteristics
        filter_conditions = [crsp_compq[col] == val for col, val in zip(columns, characteristic_values)]
        combined_filter = filter_conditions[0]
        for condition in filter_conditions[1:]:
            combined_filter &= condition

        # Apply the filter to the DataFrame
        filtered_data = crsp_compq.loc[combined_filter].copy()

        # Check if filtered_data is not empty
        if filtered_data.empty:
            print(f"No data for characteristics {characteristic_values}")
            continue

        # Compute market cap weights for each period
        filtered_data['total_cap'] = filtered_data.groupby('date')['ME_shifted'].transform('sum')
        filtered_data['weight'] = filtered_data['ME_shifted'] / filtered_data['total_cap']
        
        # Store weights data
        weights_data[characteristic_values].append(filtered_data[['date', 'PERMNO', 'weight']])

        # Compute weighted returns for each period
        filtered_data['weighted_return'] = filtered_data['RET'] * filtered_data['weight']

        # Aggregate the weighted returns by date
        aggregated_returns = filtered_data.groupby(['date'])['weighted_return'].sum().reset_index()

        #aggregated_returns = filtered_data.groupby(['date', 'period'])['weighted_return'].sum().reset_index()

        # Assign characteristic values
        for col, val in zip(columns, characteristic_values):
            aggregated_returns[col] = val

        # Append the results to the list
        results.append(aggregated_returns)

    # Concatenate all results into a single DataFrame
    if not results:
        return pd.DataFrame(), {}

    cap_weighted_returns = pd.concat(results, ignore_index=True)

    # Sort the final DataFrame by date
    cap_weighted_returns.sort_values(by='date', inplace=True)
    cap_weighted_returns.reset_index(drop=True, inplace=True)

    # Pivot the DataFrame to get one row per month and separate columns for each characteristic combination
    pivoted_returns = cap_weighted_returns.pivot_table(
        index='date',
        columns=columns,
        values='weighted_return'
    )

    # Flatten the columns
    pivoted_returns.columns = ['_'.join(map(str, col)).replace(' ', '').replace('_', '') for col in pivoted_returns.columns.values]

    # Reset index to make 'date' a column again
    pivoted_returns.reset_index(inplace=True)

    # Combine weights data into dataframes for each characteristic combination
    weights_dfs = {}
    for characteristic_values, weight_list in weights_data.items():
        if weight_list:
            weights_df = pd.concat(weight_list)
            weights_df = weights_df.pivot_table(index='date', columns='PERMNO', values='weight')
            weights_dfs[characteristic_values] = weights_df

    return pivoted_returns, weights_dfs


# List of individual characteristics
size_buckets = ['Small', 'Big']
inv_buckets = ['Conservative', 'Aggressive', 'Mid INV']
roe_buckets = ['LowROE', 'HighROE', 'Mid ROE']

# Generate combinations of characteristics
characteristics = list(itertools.product(size_buckets, inv_buckets, roe_buckets))

columns = ['Size_bucket', 'INV_bucket', 'ROE_bucket']
triple_sort, weights_dict = compute_cap_weighted_returns_monthly_rebalance(crsp_compq, characteristics, columns)

triple_sort.loc[:,'date'] = pd.to_datetime(triple_sort['date'])
triple_sort.set_index('date', inplace=True)

# Get the I/A Factor
triple_sort['I/A'] = (
    triple_sort['SmallConservativeLowROE'].fillna(0) + 
    triple_sort['SmallConservativeHighROE'].fillna(0) + 
    triple_sort['SmallConservativeMidROE'].fillna(0) + 
    triple_sort['BigConservativeLowROE'].fillna(0) + 
    triple_sort['BigConservativeHighROE'].fillna(0) + 
    triple_sort['BigConservativeMidROE'].fillna(0)
) / 6 - (
    triple_sort['SmallAggressiveLowROE'].fillna(0) + 
    triple_sort['SmallAggressiveHighROE'].fillna(0) + 
    triple_sort['SmallAggressiveMidROE'].fillna(0) + 
    triple_sort['BigAggressiveLowROE'].fillna(0) + 
    triple_sort['BigAggressiveHighROE'].fillna(0) + 
    triple_sort['BigAggressiveMidROE'].fillna(0)
) / 6

# Get the ROE Factor
triple_sort['ROE'] = (
    triple_sort['SmallConservativeHighROE'].fillna(0) + 
    triple_sort['SmallAggressiveHighROE'].fillna(0) + 
    triple_sort['SmallMidINVHighROE'].fillna(0) +
    triple_sort['BigConservativeHighROE'].fillna(0) + 
    triple_sort['BigAggressiveHighROE'].fillna(0) + 
    triple_sort['BigMidINVHighROE'].fillna(0)
) / 6 - (
    triple_sort['SmallConservativeLowROE'].fillna(0) + 
    triple_sort['SmallAggressiveLowROE'].fillna(0) + 
    triple_sort['SmallMidINVLowROE'].fillna(0) +
    triple_sort['BigConservativeLowROE'].fillna(0) + 
    triple_sort['BigAggressiveLowROE'].fillna(0) + 
    triple_sort['BigMidINVLowROE'].fillna(0)
) / 6
   

# Reindex the weights_dict so that all dataframes have the same columns
weights_dict = reindex_weights(weights_dict, PERMNOs)

# Calculate the weights for the I/A factor
weightsIA = (
    weights_dict[('Small', 'Conservative', 'LowROE')].fillna(0) +
    weights_dict[('Small', 'Conservative', 'HighROE')].fillna(0) +
    weights_dict[('Small', 'Conservative', 'Mid ROE')].fillna(0) +
    weights_dict[('Big', 'Conservative', 'LowROE')].fillna(0) +
    weights_dict[('Big', 'Conservative', 'HighROE')].fillna(0) +
    weights_dict[('Big', 'Conservative', 'Mid ROE')].fillna(0)
) / 6 - (
    weights_dict[('Small', 'Aggressive', 'LowROE')].fillna(0) +
    weights_dict[('Small', 'Aggressive', 'HighROE')].fillna(0) +
    weights_dict[('Small', 'Aggressive', 'Mid ROE')].fillna(0) +
    weights_dict[('Big', 'Aggressive', 'LowROE')].fillna(0) +
    weights_dict[('Big', 'Aggressive', 'HighROE')].fillna(0) +
    weights_dict[('Big', 'Aggressive', 'Mid ROE')].fillna(0)
) / 6

    
# Calculate the weights for the ROE factor
weightsROE = (
    weights_dict[('Small', 'Conservative', 'HighROE')].fillna(0) + 
    weights_dict[('Small', 'Aggressive', 'HighROE')].fillna(0) +
    weights_dict[('Small', 'Mid INV', 'HighROE')].fillna(0) +
    weights_dict[('Big', 'Conservative', 'HighROE')].fillna(0) +
    weights_dict[('Big', 'Aggressive', 'HighROE')].fillna(0) +
    weights_dict[('Big', 'Mid INV', 'HighROE')].fillna(0)
) / 6 - (
    weights_dict[('Small', 'Conservative', 'LowROE')].fillna(0) +
    weights_dict[('Small', 'Aggressive', 'LowROE')].fillna(0) +
    weights_dict[('Small', 'Mid INV', 'LowROE')].fillna(0) +
    weights_dict[('Big', 'Conservative', 'LowROE')].fillna(0) +
    weights_dict[('Big', 'Aggressive', 'LowROE')].fillna(0) +
    weights_dict[('Big', 'Mid INV', 'LowROE')].fillna(0)
) / 6



# Import the 18 portfolio returns series from Hou, Xue, Zhang
qfactors = pd.read_csv('/Users/Ilyas/Documents/Mémoire/qfactors.csv', sep=',')
qfactors['date'] = pd.to_datetime(qfactors[['year', 'month']].assign(day=1)) + MonthEnd(0)

# Rename the columns with explicit casting to avoid dtype issues
qfactors['rank_ME'] = np.where(qfactors['rank_ME'] == 1, 'Small', 'Big').astype(str)
qfactors['rank_IA'] = np.where(qfactors['rank_IA'] == 1, 'Conservative',
                               np.where(qfactors['rank_IA'] == 2, 'Mid INV', 'Aggressive')).astype(str)
qfactors['rank_ROE'] = np.where(qfactors['rank_ROE'] == 1, 'LowROE',
                                np.where(qfactors['rank_ROE'] == 2, 'Mid ROE', 'HighROE')).astype(str)

# Compute the returns of the I/A factor
pivot1 = qfactors.pivot(columns=['rank_IA','rank_ME', 'rank_ROE'], index='date', values='ret_vw')
HXZ_factors = pd.DataFrame(index=pivot1.index.tolist())
HXZ_factors['I/A'] = pivot1.loc[:,'Conservative'].mean(axis=1) - pivot1.loc[:,'Aggressive'].mean(axis=1)

# Compute the returns of the ROE factor
pivot2 = qfactors.pivot(columns=['rank_ROE','rank_IA','rank_ME'], index='date', values='ret_vw')
HXZ_factors['ROE'] = pivot2.loc[:,'HighROE'].mean(axis=1) - pivot2.loc[:,'LowROE'].mean(axis=1)

# Assuming triple_sort and CMA_factor are already defined DataFrames
# Merge the replicated factors with the original ones into a single dataframe
HXZcomp = pd.merge(HXZ_factors, triple_sort[['I/A','ROE']], how='inner', left_index=True, right_index=True)

# Get the correlation between the original and the replicated factor
print('ROE', f"{np.corrcoef(HXZcomp['ROE_x'], HXZcomp['ROE_y'])[1][0]:.0%}")
print('I/A', f"{np.corrcoef(HXZcomp['I/A_x'], HXZcomp['I/A_y'])[1][0]:.0%}")

# Removing unused variables
del crsp_compq_nyse, crsp_compq_shifted, nyse_data, quantiles_ROE_by_date
del size_buckets, inv_buckets, roe_buckets, characteristics, columns, weights_dict
del qfactors, pivot1, pivot2, HXZ_factors, HXZcomp


#################################################################
######################## MOMENTUM FACTOR ########################
#################################################################

# Define the lagged returns
crsp_compq['1 + RET'] = np.where((crsp_compq['RET'] < -1) & (crsp_compq['RET'] != -99), np.nan, crsp_compq['RET'] + 1)
crsp_compq.loc[:, '1 + RET'] = np.where(crsp_compq['RET'] == -99, 1, crsp_compq['1 + RET'])

def rolling_cumprod(x):
    if x.isna().any():
        return np.nan
    else:
        return np.prod(x)

crsp_compq['11-month RET'] = crsp_compq.groupby('PERMNO')['1 + RET'].rolling(window=11).apply(rolling_cumprod, raw=False).reset_index(level=0, drop=True)
crsp_compq['11-month RET'] -= 1
crsp_compq['11-month RET_lag2'] = crsp_compq.groupby('PERMNO')['11-month RET'].shift(2)

# Variables used in conditions: price at the end of month t-13 and good return at the end of month t-2
crsp_compq['PRC_lag13'] = crsp_compq.groupby('PERMNO')['PRC'].shift(13)
crsp_compq['RET_lag2'] = crsp_compq.groupby('PERMNO')['RET'].shift(2)
crsp_compq['RET_lag2'] = crsp_compq['RET_lag2'].fillna(-100)

# Compute the breakpoints for the MOM factor
mom_nyse_data = crsp_compq[(crsp_compq['NYSE'] == 1) & (crsp_compq['PRC_lag13'].notna()) & (crsp_compq['RET_lag2'] > -2 )]

quantiles_MOM_by_date = mom_nyse_data.groupby('date')['11-month RET_lag2'].quantile(q=[.3, .7]).unstack().reset_index()
quantiles_MOM_by_date.rename(columns={0.3: 'MOM breakp 0.3', 0.7: 'MOM breakp 0.7'}, inplace=True)

# Merge the quantile values back to the original DataFrame
crsp_compq = crsp_compq.merge(quantiles_MOM_by_date, on='date', how='left')

# Create a column that contains Weak if the stock has a small OP and Robust if it has a big OP
crsp_compq['MOM_bucket'] = np.where(crsp_compq['11-month RET_lag2'] <= crsp_compq['MOM breakp 0.3'], 'Low MOM', 
                                    np.where(crsp_compq['11-month RET_lag2'] > crsp_compq['MOM breakp 0.7'], 'High MOM', 
                                             np.where((crsp_compq['11-month RET_lag2'] > crsp_compq['MOM breakp 0.3']) & (crsp_compq['11-month RET_lag2'] <= crsp_compq['MOM breakp 0.7']), 'Mid MOM', 0) )) 

crsp_compq['Size_bucket'] = crsp_compq.groupby('PERMNO')['Size_bucket'].ffill(limit=11)
crsp_compq['MOM_bucket'] = crsp_compq.groupby('PERMNO')['MOM_bucket'].ffill(limit=11)

size_MOM_buckets = [('Small', 'Low MOM'), ('Small', 'High MOM'), ('Big', 'Low MOM'), ('Big', 'High MOM')]
size_MOM_columns = ['Size_bucket', 'MOM_bucket']
MOM_factor, weightsMOMdict = compute_cap_weighted_returns_monthly_rebalance(crsp_compq, size_MOM_buckets, size_MOM_columns)
MOM_factor['MOM'] = 0.5 * (MOM_factor['SmallHighMOM'] + MOM_factor['BigHighMOM']) - 0.5 * (MOM_factor['SmallLowMOM'] + MOM_factor['BigLowMOM'])

# Compute the CMA weights
weightsMOMdict = reindex_weights(weightsMOMdict, PERMNOs)
weightsMOM = 0.5 * (weightsMOMdict[('Small', 'High MOM')].fillna(0) + weightsMOMdict[('Big', 'High MOM')].fillna(0)) - 0.5 * (weightsMOMdict[('Small', 'Low MOM')].fillna(0) + weightsMOMdict[('Big', 'Low MOM')].fillna(0))


MOM_factor.set_index('date', inplace=True)

# Compare with the Momentum factor from French's website
# Load the Fama French Factor
ff_MOM_fact = pd.read_csv("/Users/Ilyas/Documents/Mémoire/F-F_Momentum_Factor.csv", skiprows=13, nrows=1167)
ff_MOM_fact.rename(columns={"Unnamed: 0":"date", 'Mom   ':'MOM'}, inplace="True")
ff_MOM_fact["date"] = pd.to_datetime(ff_MOM_fact["date"], format="%Y%m") + MonthEnd(0)
ff_MOM_fact.set_index(ff_MOM_fact["date"], drop=True, inplace=True)

# Merge the replicated factors with the original ones into a single dataframe
ffcompMOM = pd.merge(ff_MOM_fact, MOM_factor['MOM'], how='inner', left_index=True, right_index=True)


# Get the correlation between the original and the replicated factor (after 1966)
ffcompMOM66 = ffcompMOM[ffcompMOM['date']>='01/01/1966']
print('MOM', f"{np.corrcoef(ffcompMOM66['MOM_x'], ffcompMOM66['MOM_y'])[1][0]:.0%}")

# Removing unused variables
del mom_nyse_data, quantiles_MOM_by_date, size_MOM_buckets, size_MOM_columns, weightsMOMdict
del ff_MOM_fact, ffcompMOM, ffcompMOM66


#################################################################
#################### COMPUTE THE BAB FACTOR #####################
#################################################################

# Load the betas
parts = []
for i in range(1, 51):  # 50 parts
    part_df = pd.read_sql_query(f"SELECT * FROM betas_part_{i}", crsp_compustat, index_col='index')
    parts.append(part_df)
betas = pd.concat(parts, axis=1)
betas.index = pd.to_datetime(betas.index)
betas.astype(float)
betas.columns = betas.columns.astype(int)

# Import the CRSP monthly data
crsp =  pd.read_sql_query(
                            sql="SELECT * FROM crsp_m",
                            con=crsp_compustat    
                            )
crsp = crsp.astype({'shrcd':int})
crsp = crsp[crsp['shrcd'].isin(set([10,11]))].copy()
PERMNO_common =  crsp['permno'].unique().tolist()

# Keep only stocks with a share code of 10 or 11
betas = betas[betas.columns.intersection(PERMNO_common)]
del crsp

# Load the stock returns
parts = []
for i in range(1, 51):  # 50 parts
    part_df = pd.read_sql_query(f"SELECT * FROM stock_ret_part_{i}", crsp_compustat, index_col='date', parse_dates='date')
    parts.append(part_df)
stock_returns = pd.concat(parts, axis=1)
stock_returns.index = pd.to_datetime(stock_returns.index)

# Load the list of PERMNOs
PERMNOs = pd.read_sql_query("SELECT * FROM PERMNOs", crsp_compustat)
PERMNOs = PERMNOs.astype(int)
PERMNOs = PERMNOs.iloc[:,0].to_list()

# Reindex the columns to contain all the PERMNOs
betas = betas.reindex(columns=PERMNOs)

# Shrink the betas: multiply the betas by 0.6 and add 1 - 0.6
shrunk_betas = betas.mul(0.6).add(0.4)

# Compute the rank and weights of the securities
beta_rank = shrunk_betas.rank(axis=1)
median_beta_rank = beta_rank.median(axis=1, skipna=True)
rank_minus_median = beta_rank.sub(median_beta_rank, axis=0)
k = 2 / rank_minus_median.abs().sum(axis=1)
weights_beta = rank_minus_median.mul(k, axis=0)
weights_beta.fillna(0, inplace=True)

# Weights of securities in the Low and High beta portfolios
wL = weights_beta.mul(-1).where(weights_beta < 0, 0)
wH = weights_beta.where(weights_beta > 0, 0)

# Reindex the stock returns on betas' indices
stock_returns.columns = stock_returns.columns.astype(int)
stock_returns = stock_returns.reindex(columns=PERMNOs)
stock_returns = stock_returns.reindex(index=betas.index)

# Missing returns are set to 0
stock_returns.fillna(0, inplace=True)
stock_returns = stock_returns.astype(float)

# Shift the weights returns by 1 month: returns of month t+1 are multiplied by the weights from month t
shifted_wL = wL.shift(1).fillna(0)
shifted_wH = wH.shift(1).fillna(0)

shifted_wL = shifted_wL.astype(float)
shifted_wH = shifted_wH.astype(float)

# Computing the returns
ret_wL = shifted_wL.mul(stock_returns, axis=1).sum(axis=1)
ret_wH = shifted_wH.mul(stock_returns, axis=1).sum(axis=1)

# Constant to get a portfolio beta of 1
const_wL = wL.mul(shrunk_betas, axis=1).sum(axis=1).shift(1)
const_wH = wH.mul(shrunk_betas, axis=1).sum(axis=1).shift(1)
const_wL.replace(to_replace=0, value=1, inplace=True)
const_wH.replace(to_replace=0, value=1, inplace=True)


returns = pd.merge(ret_wL.rename('ret_wL'), ret_wH.rename('ret_wH'), how='inner', left_index=True, right_index=True)
returns = returns.loc['1966-02-28 ':].copy()

# Import the monthly risk-free rate from French's website
ff_original_fact = pd.read_csv("/Users/Ilyas/Documents/Mémoire/_FF 5 Factors_2x3.csv", skiprows=3, nrows=729)
ff_original_fact.rename(columns={"Unnamed: 0":"date", 'RF':'TBill'}, inplace="True")
ff_original_fact["date"] = pd.to_datetime(ff_original_fact["date"], format="%Y%m") + MonthEnd(0)
ff_original_fact.set_index(ff_original_fact["date"], drop=True, inplace=True)

returns = returns.merge(ff_original_fact.TBill, how='inner', left_index=True, right_index=True)
returns = returns.merge(const_wL.rename('const_wL'), how='inner', left_index=True, right_index=True)
returns = returns.merge(const_wH.rename('const_wH'), how='inner', left_index=True, right_index=True)

returns.loc[:,'TBill'] = returns.TBill.div(100)

# Compute the returns of the BAB factor
returns['BAB'] = (returns['ret_wL'] - returns['TBill']).div(returns['const_wL'], axis=0) - (returns['ret_wH'] - returns['TBill']).div(returns['const_wH'], axis=0)

# Get the weights of the BAB factor
betas_weights = shifted_wL.div(returns['const_wL'], axis=0) - shifted_wH.div(returns['const_wH'], axis=0)
betas_weights.index.rename('date', inplace=True)

# Compare with the original BAB factor
FP_factor = pd.read_excel('/Users/Ilyas/Documents/Mémoire/Betting Against Beta Equity Factors Monthly.xlsx', sheet_name=0, skiprows=18,)
FP_factor = FP_factor[['DATE','USA']].copy()
FP_factor.rename(columns={'DATE':'date','USA':'BAB'}, inplace=True)

FP_factor.loc[:,'date'] = pd.to_datetime(FP_factor['date'])
FP_factor.set_index('date',inplace=True)
FP_factor.astype(float)

FP_factor = FP_factor.merge(returns['BAB'].iloc[1:].astype(float), how='inner', left_index=True, right_index=True)
print('BAB', f"{np.corrcoef(FP_factor['BAB_x'], FP_factor['BAB_y'])[1][0]:.0%}")


# Removing unused variables
del shrunk_betas, beta_rank, rank_minus_median, k, weights_beta, wL, wH, shifted_wL, shifted_wH
del ret_wL, ret_wH, const_wL, const_wH, FP_factor



#################################################################
######################## SAVE THE RESULTS #######################
#################################################################


# Split the columns into 50 parts
n_splits = 50
columns_split = np.array_split(stock_returns.columns, n_splits)

# Save each part to the SQL database
for i, cols in enumerate(columns_split, 1):
    part_df = stock_returns[cols]
    part_df.to_sql(name=f'stock_ret_part_{i}', 
              con=crsp_compustat, 
              if_exists="replace") 
        
# Function to save the dataframes to sql
def save_as_sql(df, name):
    # Split the columns into 50 parts
    n_splits = 50
    columns_split = np.array_split(df.columns, n_splits)
    
    # Save each part to the SQL database
    for i, cols in enumerate(columns_split, 1):
        part_df = df[cols]
        part_df.to_sql(name=f'weights_{name}_part_{i}', 
                  con=crsp_compustat, 
                  if_exists="replace") 


weightsMarket = weightsMarket.reindex(columns=PERMNOs)

save_as_sql(weightsMarket, 'Market')
save_as_sql(weightsSMB, 'SMB')
save_as_sql(weightsHML, 'HML')
save_as_sql(weightsRMW, 'RMW')
save_as_sql(weightsCMA, 'CMA')
save_as_sql(weightsIA, 'IA')
save_as_sql(weightsROE, 'ROE')
save_as_sql(weightsMOM, 'MOM')
save_as_sql(betas_weights, 'BAB')


# Combine the factors in a single dataframe
factors = pd.merge(market_factor, size_factor['SMB'], how='inner', on='date')
factors = pd.merge(factors, HML_factor['HML'], how='inner', on='date')
factors = pd.merge(factors, RMW_factor['RMW'], how='inner', on='date')
factors = pd.merge(factors, CMA_factor['CMA'], how='inner', on='date')
factors = pd.merge(factors, MOM_factor['MOM'], how='inner', on='date')
factors = pd.merge(factors, triple_sort[['I/A','ROE']], how='inner', on='date')
factors = pd.merge(factors, returns['BAB'], how='inner', right_index=True, left_index=True)

(factors.to_sql(name="factors", 
          con=crsp_compustat, 
          if_exists="replace",
          index=True)
)



#################################################################
##################### COMPUTE MARKET RETURNS ####################
#################################################################

# Get the market returns
conn = sqlite3.connect("/Users/Ilyas/Documents/Mémoire/crsp_compustat.sqlite")

# Load factor returns and market volatility
mret = pd.read_sql_query("SELECT permno, date, prc, shrout, ret FROM crsp_m", conn)

# Rename columns 
ccols = {'permno': 'PERMNO',
         'ret' : 'RET',
         'prc' : 'PRC',
         'shrout' : 'SHROUT'}
mret = mret.rename(columns = ccols)

# Dictionary to type 32bit
dictotype32 = {'PERMNO': np.int32,
               'PRC': float,
               'SHROUT': float,
               'RET': float}

# Convert the variables to the required type
mret = mret.astype(dictotype32)

# Returns that are -66, -77, -88, -99 are mapped to null
mret['RET'] = np.where(mret['RET'] < -1, np.nan, mret['RET'])

# Convert the date to a datetime format
mret['date'] = pd.to_datetime(mret['date'])

# Line up date to be end of month
mret['date'] = mret['date'] + MonthEnd(0)

# Ensure that price data is positive
mret['PRC'] = np.abs(mret['PRC'])

# Compute the market cap at the PERMNO level, and set it to NaN if it is 0
mret['CAP'] = mret['PRC'] * mret['SHROUT']
mret['CAP'] = np.where(mret['CAP']==0, np.nan, mret['CAP'])

# Drop duplicates
mret.drop_duplicates(subset=['date', 'PERMNO'], ignore_index=True, inplace=True)
mret = mret.sort_values(by=['PERMNO', 'date']).reset_index(drop=True)

# Shift the market cap to get the weights from the previous month
mret['CAP_SHIFTED'] = mret.groupby('PERMNO')['CAP'].shift(1)

# Compute the cap-weighted returns
ME_shifted_total = mret.groupby('date')['CAP_SHIFTED'].apply(lambda x: x.sum())
ME_shifted_total.name = 'ME_shifted_total'
mret = mret.merge(ME_shifted_total, how='left', left_on='date', right_index=True)

# Calculate weights
mret['weight'] = mret['CAP_SHIFTED'].div(mret['ME_shifted_total'])

# Calculate portfolio returns
mret['Mkt Ret'] = mret['RET'] * mret['weight']

# Compute the portfolio return for each month
market_returns = mret.groupby('date')['Mkt Ret'].sum()

# Save the results
conn = sqlite3.connect("/Users/Ilyas/Documents/Mémoire/market.sqlite")
market_returns.to_sql(name='mret', con=conn, if_exists="replace", index='date')
