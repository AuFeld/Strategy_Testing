import pandas as pd
import os

SEQ_LEN = 60
# sequence is 60 mins long
FUTURE_PERIOD_PREDICT = 3
# 1 period = 1 minute, eg 3 periods = 3 mins
RATIO_TO_PREDICT = 'LTC-USD'

"""
Establishing a realtionship for the classify function. 
It states that if the future value that we are predicting is greater then the current value, 
then we want to indetify that future predicted value with a 1. 
If not, it will be assigned with a 0.
"""
def classify(current, future):
    if float(future) > float(current):
        return 1
    else: 
        return 0

main_df = pd.DataFrame()

names = ['time', 'low', 'high', 'open', 'close', 'volume']
ratios = ['BTC-USD', 'LTC-USD', 'ETH-USD', 'BCH-USD']

"""
Makes new dataframe that incorporates all crypto currencies
"""
for ratio in ratios:
    dataset = f'crypto_data/{ratio}.csv'

    df = pd.read_csv(dataset, names=names)
    df.rename(columns={'close': f'{ratio}_close', 
                       'volume': f'{ratio}_volume'}, 
                       inplace=True)

    df.set_index('time', inplace=True)
    df = df[[f'{ratio}_close', f'{ratio}_volume']]

    if len(main_df) == 0:
        main_df = df
    else: 
        main_df = main_df.join(df)

main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)

main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))

print(main_df[[f'{RATIO_TO_PREDICT}_close', 'future', 'target']].head())