import bs4 as bs
import requests
import yfinance as yf
import datetime
import csv
import pandas as pd

resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = bs.BeautifulSoup(resp.text, 'lxml')
table = soup.find('table', {'class': 'wikitable sortable'})
tickers = []
for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text
    tickers.append(ticker)

tickers = [s.replace('\n', '') for s in tickers]
start = datetime.datetime(2019,1,1)
end = datetime.datetime(2019,7,17)
data = yf.download(tickers, start=start, end=end)
print(data)

with open('/home/chase/repos/NN_Predicting/sp500/sp500.csv', 'rb') as f:
  data = list(csv.reader(f))

import collections
counter = collections.defaultdict(int)
for row in data:
    counter[row[0]] += 1

writer = csv.writer(open("/home/chase/repos/NN_Predicting/sp500/sp500.csv", 'w'))
for row in data:
    if counter[row[0]] >= 4:
        writer.writerow(row)

df = pd.read_csv('/home/chase/repos/NN_Predicting/sp500/sp500.csv', delim_whitespace=True)
print(df)