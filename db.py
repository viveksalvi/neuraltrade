
import sqlite3 

con = sqlite3.connect("./data/trade.db")

cur = con.cursor()

sql = """
    CREATE TABLE asset_prices(
        asset_name TEXT NOT NULL,
        created_at TEXT NOT NULL,
        price_high REAL,
        price_low REAL,
        price_open REAL,
        price_close REAL,
        PRIMARY KEY (asset_name, created_at)
    );
"""

cur.execute(sql)
con.commit()
con.close()


