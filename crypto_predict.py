import pandas as pd
import os
from sklearn import preprocessing
from collections import deque
import random
import numpy as np
import time
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Activation, Dense, Dropout, LSTM, BatchNormalization, LeakyReLU 
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint 

'''
Variables:
How many batches? Try smaller batches if you're getting an OOM error.
OOM = Out of Memory.
SEQ_LEN = how long of a preceeding sequence to collect for RNN.
Sequence Length is 60 mins long. 
EPOCHS = how many passes through our data.
'''
SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
# 1 period = 1 minute, eg 3 periods = 3 mins
RATIO_TO_PREDICT = 'LTC-USD'
EPOCHS = 10
BATCH_SIZE = 64 
NAME = f'{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}'

'''
Establishing a realtionship with the classify function. 
It states that if the future value that we are predicting is greater then the current value, 
then we want to indetify that future predicted value with a 1. 
If not, it will be assigned with a 0.
'''
def classify(current, future):
    if float(future) > float(current):
        return 1
    else: 
        return 0

'''
Preprocess function
'''
def preprocess_df(df):
    df = df.drop('future', 1)

    for col in df.columns:
        if col != 'target':
            '''
            Normalize the data
            '''
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            '''
            Scale the data
            '''
            df[col] = preprocessing.scale(df[col].values)
    
    df.dropna(inplace=True)

    sequential_data = []
    '''
    As new data is added to this list (a max of 60 via SEQ_LEN), 
    deque will drop the old data
    '''
    prev_days = deque(maxlen=SEQ_LEN)
    
    for i in df.values:
        '''
        Appending each value in the lists of lists (each of the columns),
        up to the last i, but not including our target feature.
        '''
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
    
    random.shuffle(sequential_data)
    '''
    Balance the data
    '''
    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])
    
    random.shuffle(buys)
    random.shuffle(sells)
    '''    
    what's that minimum of the two lists, buys and sells?
    '''
    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys+sells
    random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)
    
    return np.array(X), y


main_df = pd.DataFrame()

names = ['time', 'low', 'high', 'open', 'close', 'volume']
ratios = ['BTC-USD', 'LTC-USD', 'ETH-USD', 'BCH-USD']

'''
Make new dataframe that incorporates all crypto currencies
'''
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
#print(main_df[[f'{RATIO_TO_PREDICT}_close', 'future', 'target']].head())

times = sorted(main_df.index.values)
last_5pct = times[-int(0.05*len(times))]

val_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

train_x, train_y, = preprocess_df(main_df)
val_x, val_y = preprocess_df(val_main_df)

'''
Verify data split
'''
print(f'train data: {len(train_x)}, val: {len(val_x)}')
print(f'Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}')
print(f'VALIDATION Dont buys: {val_y.count(0)}, buys: {val_y.count(1)}')

'''
RNN Architecture
''' 
model = Sequential(name='Crypto_Model')

model.add(Dense(128, input_shape=(train_x.shape[1:]), activation='relu'))

model.add(Dense(128, activation='relu'))
model.add(LeakyReLU(alpha=0.05))
model.add(Dense(128, activation='relu'))

model.add(Dense(128, activation='softmax'))
'''

model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Droupout(0.2))

model.add(Dense(2, activation="softmax"))
'''
opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

'''
Compile Model
'''
model.compile(loss='binary_crossentropy', 
              optimizer=opt,
              metrics=['acc', 'binary_crossentropy', 'cosine_proximity'])

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

'''
Unique file name that will include the epoch and
the validation accuracy for that epoch.
Checkpoint only saves the best ones.
'''
filepath = 'RNN_Final-{epoch:02d}-{val_acc:.3f}'
checkpoint = ModelCheckpoint('models/{}.model'.format(filepath, 
                                                      monitor='val_acc', 
                                                      verbose=1, 
                                                      save_best_only=True, 
                                                      mode='max'))

'''
Train the model
'''
train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
val_x = np.asarray(val_x)
val_y = np.asarray(val_y)

history = model.fit(train_x, train_y,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(val_x, val_y),
                    callbacks=[tensorboard, checkpoint])

print(train_x.shape)
print(train_y.shape)