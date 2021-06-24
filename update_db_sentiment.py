import pandas as pd
from datetime import datetime, timedelta
import data.db_interaction
from data.reddit_retrieve import *
from data.gtrends_retrieve import *

DI = data.db_interaction.DatabaseInteraction()

# Query latest date in posts table, pass to scraping function to get post and comment data
latest_date = DI.query_latest('posts', 'date')

posts_df, comments_df = scrape_bitcoin_reddit(latest_date)

today = datetime.datetime.now()

# Get Google Trends data for a given keyword and date range
today = datetime.datetime.now()

gtrends_df = get_gtrends(keyword='bitcoin',
                         start_year=2015,
                         start_month=1,
                         end_year=today.year,
                         end_month=today.month
                        )

# Save Reddit data to postgres db
DI.df_to_db('posts', posts_df, pd_if_exists='append')
DI.df_to_db('comments', comments_df, pd_if_exists='append')

# Save Reddit and Google data to postgres db
DI.df_to_db('gtrends', gtrends_df, pd_if_exists='replace')