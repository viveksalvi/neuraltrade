import requests
import json
import sqlite3 
import datetime;


def prepare_for_db(resp, asset_name):
    raw_data = json.loads(resp.content)
    sql_data = list()
    for each_d in raw_data["data"]:
        sql_data.append(
            tuple((asset_name, each_d['created_at'],each_d['high'],each_d['low'],each_d['open'],each_d['close']))
        )
    return sql_data


def write_to_db(data:list()):
    con = sqlite3.connect("./data/trade.db")
    cur = con.cursor()
    sql_insert = """
        INSERT INTO asset_prices values (?, ?, ?, ?, ?, ?)
        ON CONFLICT (asset_name, created_at)
        DO NOTHING
    """
    cur.executemany(sql_insert, data)
    con.commit()
    con.close()


def injest():
    ct = datetime.datetime.now()
    ct = ct - datetime.timedelta(hours=4)

    assets = ['Z-CRY/IDX', 'ALTHUOB/BTC-CXDX']
    for i in range(0, 10):
        year = ct.year
        month = str(ct.month) if len(str(ct.month)) == 2 else f'0{str(ct.month)}'
        day = str(ct.day) if len(str(ct.day)) == 2 else f'0{str(ct.day)}' # ct.day
        hour = str(ct.hour) if len(str(ct.hour)) == 2 else f'0{str(ct.hour)}' # ct.hour
        
        for each_asset in assets:
            resp = requests.get(f"https://api.binomo.com/platform/candles/{each_asset}/{year}-{month}-{day}T{hour}:00:00/5?locale=en")
            data = prepare_for_db(resp, each_asset)

            print(f"{ct} | {len(data)} | {each_asset}")
            if len(data) > 0:
                write_to_db(data)
        
        ct = ct - datetime.timedelta(hours=1)


def update_thread():
    while True:
        injest()
        import time
        time.sleep(2)
import threading

x = threading.Thread(target=update_thread)
x.start()
x.join()