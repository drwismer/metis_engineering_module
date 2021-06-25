# Data Engineering Project - The State of Bitcoin

For this bootcamp project, I was tasked with building a modularized data pipeline. I decided to build a pipeline that used two API's (jesse.trade and Glassnode) to gather information about Bitcoin. This data was manipulated to calculate a wide variety of indicators to help a trader make buy or sell decisions. These indicators include Ichimoku Cloud, Bollinger Bands, MACD, RSI, and a variety of on-chain metrics. 

In addition to displaying the data I pulled in an organized fashion, I also developed some different looks that I've not come across during my time on, for example, TradingView. This is not to say that these are brand new ideas, but the visualizations are novel to me. Some new (to me) visualizations I built include:
- A different look at the size and orientation of the Ichimoku cloud over time, with a built-in kumo twist indicator.
- Different looks at the size of Bollinger Bands over time, as well as visualizing the position of the closing Bitcoin price within that band.
- On-chain metrics vs. price. Rather than displaying the on-chain metric (for example, Net Unrealized Profit/Loss) as it's own line within the plot, I instead created colored panes to indicate when the metric was as a certain level. This allows for a quick scan of historical levels that precipitated strong price movements.
- Divergences. Divergences between price and metrics like volume, RSI, and MACD are common trading tools, but it can be difficult to spot these divergences. I aimed to visualize divergences by calculating the slope of the regression line for a given lookback period (15 days, 30 days, 60 days, etc.). By plotting the time-series data of the regression slope over time, you can spot divergences when the two metrics (for example, Price and Volume) are on opposite signs of zero, represented by a dashed midline. Similar to the concept for on-chain metrics, I also built a visualization that shows colored red areas for bearish divergences and colored green areas for bullish divergences.

Finally, I began tuning and modeling a Random Forest Classification model for predicting whether the price of Bitcoin would be higher or lower one week from the prediction. I engineered several features, including interactions between on-chain metrics, moving averages (and their regression slopes), and various ratios, particularly related to Ichimoku Cloud metrics. You can see my work in the modeling notebook [here](https://github.com/drwismer/metis_engineering_module/blob/main/Bitcoin%20Random%20Forest%20Classifier.ipynb).

All the project write-ups and modeling I describe above can be found in the main folder of this repository. The data retrieval modules can be found in the data folder. For a full write-up of this project, including screenshots of the final dashboard, please [click here](https://github.com/drwismer/metis_engineering_module/blob/main/project_writeup_state_of_bitcoin.md). 
