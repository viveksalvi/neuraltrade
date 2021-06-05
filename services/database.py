import sqlite3 
import pandas as pd


def get_last_n_prices_by_asset(n, asset_name):
    con = sqlite3.connect("./data/trade.db")
    sql = f"""
        select * from asset_prices ap 
        where asset_name = "{asset_name}"
        order by created_at DESC
        limit {n}
    """
    data = pd.read_sql(sql, con)
    return data
