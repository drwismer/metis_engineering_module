# Project Proposal - State of Bitcoin Dashboard

For the Data Engineering module project, I will create a pipeline that produces a daily dashboard that provides information summarizing the state of Bitcoin. Using jesse.trade, I will pull price data, including trading volume and candle data (opening, closing, high, and low prices). I will also use the Glassnode API to pull on-chain metrics. I will manipulate the collected data to calculate a variety of price indicators and additional novel features. I will organize the data in a daily dashboard that can be launched in a web application, likely using Bokeh. I will also provide a one week price prediction (up vs. down) using a random forest classification model that I have been developing.

This dashboard will function as a one-stop shop for reviewing the state of Bitcoin as a cryptocurrency asset. Stretch goals / future work may include including sentiment-related information, including Reddit threads, Google Trends data, or a stream of recently posted articles.

### Proposed Pipeline:

The pipeline below will be modularized and ideally run through a single Python script.

1. Data Collection
    - Capture candle and volume data with **jesse.trade**.
    - Capture on-chain data with **Glassnode API**.
    - ***Stretch Goal***: Obtain Reddit and/or news articles data.
2. Data Manipulation
    - Calculate price indicators (MACD, RSI, Ichimoku Cloud, etc.).
3. Data Storage
    - Store collected and calculated data in PostgreSQL database.
4. Modeling
    - Generate price prediction (up vs. down) in a random forest model.
5. Presentation
    - Organize information on a Bokeh dashboard that can be launched as a web application.
