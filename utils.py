import numpy as np
import sqlite3
import pandas as pd

# FUNCTION TO CREATE 1D DATA INTO TIME SERIES DATASET
def new_dataset(dataset, step_size):
	data_X, data_Y = [], []
	for i in range(len(dataset)-step_size-1):
		a = dataset[i:(i+step_size), 0]
		data_X.append(a)
		data_Y.append(dataset[i + step_size, 0])
	return np.array(data_X), np.array(data_Y)


def read_data_by_asset_name(asset_name):
    print("================================= READING DATASET =================================")
    con = sqlite3.connect('./data/trade.db')
    sql = f"""
        SELECT price_open as Open, price_high as High, price_low as Low, price_close as Close
        FROM asset_prices ap 
        WHERE asset_name = '{asset_name}'
        --AND created_at > '2021-05-14T17:50:35.000000Z'
        ORDER BY created_at asc
    """
    dataset = pd.read_sql_query(sql, con)
    print("============================= READING DATASET COMPLETED =============================")    
    return dataset


def df_to_json(df):
    return df.to_json(orient="records")

