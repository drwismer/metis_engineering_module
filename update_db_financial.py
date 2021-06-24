from jesse import research
import data.glassnode_retrieve
import data.db_interaction
from data.indicators import *
import data.time_series_functions
from datetime import datetime, timedelta

# Get latest candle date from database
DI = data.db_interaction.DatabaseInteraction()
timestamp = DI.query_latest('candle', 'timestamp')/1000

# Get candle data
exchange ='Coinbase'
coin = 'BTC-USD'
period = '1D'
start = '2015-08-15'
end = datetime.strftime(datetime.fromtimestamp(timestamp), '%Y-%m-%d')

research.init()
candles = research.get_candles(exchange, coin, period, start, end)

# Get glassnode data
start_date = '2013-01-01'

GN = data.glassnode_retrieve.GlassnodeRetrieve()
glassnode_df = GN.get_glassnode_data(start_date)

# Create pandas dataframes
cloud_df = candles_to_ichimoku(candles)
volume_df = candles_to_volume(candles)
macd_df = candles_to_macd(candles)
rsi_df = candles_to_rsi(candles)
bbands_df = candles_to_bbands(candles)

# Load dataframes to postgres db
DI.df_to_db('glassnode', glassnode_df)
DI.df_to_db('ichimoku', cloud_df)
DI.df_to_db('volume', volume_df)
DI.df_to_db('macd', macd_df)
DI.df_to_db('rsi', rsi_df)
DI.df_to_db('bbands', bbands_df)