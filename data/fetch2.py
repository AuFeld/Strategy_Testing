import pandas as pd
import yfinance as yf
import numpy as np
import csv as csv

# Year - Month - Day
start = '2020-01-01'
end = '2020-05-22'
interval = '1d'

df = pd.read_csv('/home/chase/repos/NN_Predicting/sp500/S&P500-Symbols.csv', names=['Symbol'])
inputfile = df['Symbol']

inputm = []

with open(inputfile, "rb") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        inputm.append(row)

'''
with open('/home/chase/repos/NN_Predicting/sp500/S&P500-Symbols.csv', 'rb') as f_input:
    csv_input = csv.reader(f_input)
    header = next(csv_input)
    data = zip(*[map(int, row) for row in csv_input])

print(data.head())

df = pd.DataFrame.from_csv('/home/chase/repos/NN_Predicting/sp500/S&P500-Symbols.csv', names=['Symbols'])
rows = df.apply(lambda x: x.tolist(), axis=1)

tickers = csv_array[0]


list = np.asarray(inputm)

data = yf.download(list, start=start, end=end, interval=interval, auto_adjust=True)
'''
print(inputm)