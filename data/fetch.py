import yfinance as yf

# TO DO: import every sympol from the s&P500

'''
Preset variable parameters for yfinance
'''
# y - m - d
start = '2010-01-01'
end = '2020-05-22'
interval = '1d'
#tickers = yf.Tickers()

data = yf.download(tickers='SPY MSFT IBM AMZN GOOG AAPL', interval=interval, start='2010-01-01', group_by='ticker', auto_adjust=True)

print(data.head())