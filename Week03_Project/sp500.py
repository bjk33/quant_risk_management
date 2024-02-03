import pandas as pd
import yfinance as yf

# Get ticker list
tickers = pd.read_excel("holdings-daily-us-en-spy.xlsx", sheet_name="top100").iloc[:,1].tolist()

tickers = ['SPY'] + tickers

df_return = pd.DataFrame()
# Download historical data for each ticker
for ticker in tickers:
    df = yf.download(ticker,start='2020-01-01', end='2023-02-10', progress=False)
    # Calculate daily return
    df[ticker] = df['Adj Close'].pct_change()
    df_return = pd.concat([df_return,df[ticker]],axis=1)

#Save the last 60 days' return to file
df_return = df_return.iloc[-250:,:]
df_return.to_csv("DailyReturn.csv")

