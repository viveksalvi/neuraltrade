from flask import Flask
import services.database as db_service
import utils
from flask_cors import CORS, cross_origin

ASSET_NAME = "ALTHUOB/BTC-CXDX"

app = Flask(__name__)
cors = CORS(app, resources={r"/services": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def index():
    return "Hello World!"

@app.route("/services/last/<n>")
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def last_n_prices(n=1000):
    df = db_service.get_last_n_prices_by_asset(n, ASSET_NAME)
    return utils.df_to_json(df)


app.run("localhost", 1234)
