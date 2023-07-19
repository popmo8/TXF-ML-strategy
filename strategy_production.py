from collections import defaultdict, deque
from shioaji import BidAskFOPv1, Exchange
import shioaji as sj
import datetime
import pandas as pd
import talib as ta
import time
from math import ceil
import pytrader as pyt
##########
import pickle
import numpy as np
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Flatten
from tensorflow.keras.models import Sequential
##########

api_key=''  # 自行填入
secret_key='' # 自行填入

api = sj.Shioaji(simulation=True)
accounts = api.login(
    api_key=api_key,
    secret_key=secret_key
)


trader = pyt.pytrader(strategy='109072142', api_key=api_key, secret_key=secret_key)
# 設定商品
trader.contract('TXF')

contract = min(
    [
        x for x in api.Contracts.Futures.TXF
        if x.code[-2:] not in ["R1", "R2"]
    ],
    key=lambda x: x.delivery_date
)

msg_queue = defaultdict(deque)
api.set_context(msg_queue)


@api.on_bidask_fop_v1(bind=True)
def quote_callback(self, exchange: Exchange, bidask: BidAskFOPv1):
    # append quote to message queue
    self['bidask'].append(bidask)


api.quote.subscribe(
    contract,
    quote_type=sj.constant.QuoteType.BidAsk,
    version=sj.constant.QuoteVersion.v1
)

time.sleep(2.5)

##########
with open('scaler_v6.pkl', 'rb') as f:
    scaler_loaded = pickle.load(f)

input_shape = (10, 27)
model_checkoint = Sequential()
model_checkoint.add(LSTM(256, activation="relu", input_shape=input_shape, return_sequences=True))
model_checkoint.add(LSTM(256, return_sequences=True))
model_checkoint.add(TimeDistributed(Dense(1)))
model_checkoint.add(Flatten())
model_checkoint.add(Dense(5,activation='linear'))
model_checkoint.add(Dense(1, activation="sigmoid"))
model_checkoint.load_weights('model_v6.h5')

history = []
up_threshold = 0.61596316
down_threshold = 0.3671736
time_window = 10
columns_input = ['vol_mom1', 'vol_mom2', 'vol_mom3', 'vol_mom4', 'vol_mom5', 'MACD', 'RSI', 'low_diff1',
                'low_diff2', 'low_diff3', 'low_diff4', 'low_diff5', 'close_mom1', 'close_mom2', 'close_mom3', 'close_mom4',
                 'close_mom5', 'close_diff1', 'close_diff2', 'close_diff3', 'close_diff4', 'close_diff5', 'high_diff1',
                 'high_diff2', 'high_diff3', 'high_diff4', 'high_diff5']
columns_to_scale = ['vol_mom1', 'vol_mom2', 'vol_mom3', 'vol_mom4', 'vol_mom5', 'MACD', 'RSI', 'close_mom1',
                    'close_mom2', 'close_mom3', 'close_mom4', 'close_mom5']
take_point = 40
stop_point = 20
adjust_point = 30
fee = 116
##########

while datetime.datetime.now().time() < datetime.time(8, 45):
    pass

# get maximum strategy kbars to dataframe, extra 30 it's for safety
bars = 65 + 30

# since every day has 60 kbars (only from 8:45 to 13:45), for 5 minuts kbars
days = ceil(bars/60)

df_5min = []
while(len(df_5min) < bars):
    kbars = api.kbars(
        contract=api.Contracts.Futures.TXF.TXFR1,
        start=(datetime.date.today() -
               datetime.timedelta(days=days)).strftime("%Y-%m-%d"),
        end=datetime.date.today().strftime("%Y-%m-%d"),
    )
    df = pd.DataFrame({**kbars})
    df.ts = pd.to_datetime(df.ts)
    df = df.set_index('ts')
    df.index.name = None
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.between_time('08:44:00', '04:00:01')
    df_5min = df.resample('5T', label='right', closed='right').agg(
        {'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
         })
    df_5min.dropna(axis=0, inplace=True)
    days += 1

ts = datetime.datetime.now()

print("before looping")
# print("df_5min:", df_5min)

while datetime.datetime.now().time() < datetime.time(13, 40):
    time.sleep(1)

    # this place can add stop or limit order
    if(len(trader.position()) == 0):
        self_position = 'None'
    else:
        self_position = 'Buy' if trader.position()['is_long'] else 'Sell'
    
    if self_position == 'Buy':    
        if trader.position()['pnl']+fee <= -200 * stop_point: # 15 point stop loss
            trader.sell(size = 1)
            print("close stop loss", trader.trades[-1]['exit_time'])
            print("entry price:", trader.trades[-1]['entry_price'], "exit price:", trader.trades[-1]['exit_price'], "pnl:", trader.trades[-1]['pnl'])
        elif trader.position()['pnl']+fee >= 200 * take_point: # 40 point take profit
            trader.sell(size = 1)
            print("close take profit", trader.trades[-1]['exit_time'])
            print("entry price:", trader.trades[-1]['entry_price'], "exit price:", trader.trades[-1]['exit_price'], "pnl:", trader.trades[-1]['pnl'])
    elif self_position == 'Sell':    
        if trader.position()['pnl']+fee <= -200 * stop_point: # 15 point stop loss
            trader.buy(size = 1)
            print("close stop loss", trader.trades[-1]['exit_time'])
            print("entry price:", trader.trades[-1]['entry_price'], "exit price:", trader.trades[-1]['exit_price'], "pnl:", trader.trades[-1]['pnl'])
        elif trader.position()['pnl']+fee >= 200 * take_point: # 40 point take profit
            trader.buy(size = 1)
            print("close take profit", trader.trades[-1]['exit_time'])
            print("entry price:", trader.trades[-1]['entry_price'], "exit price:", trader.trades[-1]['exit_price'], "pnl:", trader.trades[-1]['pnl'])

    # local time > next kbars time
    if(datetime.datetime.now() >= ts):

        kbars = api.kbars(
            contract=api.Contracts.Futures.TXF.TXFR1,
            start=datetime.date.today().strftime("%Y-%m-%d"),
            end=datetime.date.today().strftime("%Y-%m-%d"),
        )
        df = pd.DataFrame({**kbars})
        df.ts = pd.to_datetime(df.ts)
        df = df.set_index('ts')
        df.index.name = None
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.between_time('08:44:00', '04:00:01')
        df = df.resample('5T', label='right', closed='right').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'})
        df.dropna(axis=0, inplace=True)
        df_5min.update(df)
        to_be_added = df.loc[df.index.difference(df_5min.index)]
        df_5min = pd.concat([df_5min, to_be_added])
        ts = df_5min.iloc[-1].name.to_pydatetime()

        # next kbar time update and local time < next kbar time
        if (datetime.datetime.now().minute != ts.minute):
            
            df_5min = df_5min[:-1]
            # print(df_5min)

            self_macd, self_macd_signal, self_macd_hist = ta.MACD(df_5min['Close'])
            self_rsi = ta.RSI(df_5min['Close'], 14)

            ##################
            vol_mom1 = df_5min['Volume'][-1] - df_5min['Volume'][-2]
            vol_mom2 = df_5min['Volume'][-1] - df_5min['Volume'][-3]
            vol_mom3 = df_5min['Volume'][-1] - df_5min['Volume'][-4]
            vol_mom4 = df_5min['Volume'][-1] - df_5min['Volume'][-5]
            vol_mom5 = df_5min['Volume'][-1] - df_5min['Volume'][-6]
            macd = self_macd_hist[-1]
            rsi = self_rsi[-1]
            low_diff1 = 1 if df_5min['Low'][-1] > df_5min['Low'][-2] else -1
            low_diff2 = 1 if df_5min['Low'][-2] > df_5min['Low'][-3] else -1
            low_diff3 = 1 if df_5min['Low'][-3] > df_5min['Low'][-4] else -1
            low_diff4 = 1 if df_5min['Low'][-4] > df_5min['Low'][-5] else -1
            low_diff5 = 1 if df_5min['Low'][-5] > df_5min['Low'][-6] else -1
            close_mom1 = df_5min['Close'][-1] - df_5min['Close'][-2]
            close_mom2 = df_5min['Close'][-1] - df_5min['Close'][-3]
            close_mom3 = df_5min['Close'][-1] - df_5min['Close'][-4]
            close_mom4 = df_5min['Close'][-1] - df_5min['Close'][-5]
            close_mom5 = df_5min['Close'][-1] - df_5min['Close'][-6]
            close_diff1 = 1 if df_5min['Close'][-1] > df_5min['Close'][-2] else -1
            close_diff2 = 1 if df_5min['Close'][-2] > df_5min['Close'][-3] else -1
            close_diff3 = 1 if df_5min['Close'][-3] > df_5min['Close'][-4] else -1
            close_diff4 = 1 if df_5min['Close'][-4] > df_5min['Close'][-5] else -1
            close_diff5 = 1 if df_5min['Close'][-5] > df_5min['Close'][-6] else -1
            high_diff1 = 1 if df_5min['High'][-1] > df_5min['High'][-2] else -1
            high_diff2 = 1 if df_5min['High'][-2] > df_5min['High'][-3] else -1
            high_diff3 = 1 if df_5min['High'][-3] > df_5min['High'][-4] else -1
            high_diff4 = 1 if df_5min['High'][-2] > df_5min['High'][-5] else -1
            high_diff5 = 1 if df_5min['High'][-3] > df_5min['High'][-6] else -1
            data = [[vol_mom1, vol_mom2, vol_mom3, vol_mom4, vol_mom5, macd, rsi, low_diff1, low_diff2, low_diff3, low_diff4, low_diff5,
                     close_mom1, close_mom2, close_mom3, close_mom4, close_mom5, close_diff1, close_diff2, close_diff3, close_diff4, close_diff5,
                     high_diff1, high_diff2, high_diff3, high_diff4, high_diff5]]
            # print("data before: ", data)
            tmp = pd.DataFrame(data, columns=columns_input)
            tmp[columns_to_scale] = scaler_loaded.transform(tmp[columns_to_scale])
            # print("data:", list(np.array(tmp)[0]))
            history.append(list(np.array(tmp)[0]))
            ##################

            time_condition1 = datetime.datetime.now().time() < datetime.time(13, 30)
            time_condition2 = datetime.datetime.now().time() > datetime.time(9, 15)
            time_condition3 = datetime.datetime.now().time() >= datetime.time(13, 30)

            if(len(trader.position()) == 0):
                self_position = 'None'
            else:
                self_position = 'Buy' if trader.position()['is_long'] else 'Sell'

            if (self_position == 'None' and len(history) >= time_window and time_condition1 and time_condition2):
                time_index = -1 * time_window
                x_in = [history[time_index:]]
                # print("x_in: ", len(x_in))
                y = model_checkoint.predict(x_in, verbose=0)
                print("label: ", y, datetime.datetime.now(), "\n")
                if (y[0][0] > up_threshold):    
                    trader.buy(size = 1)
                    take_point = 40
                    stop_point = 20
                    adjust_point = 30
                    in_price = df_5min['Close'][-1]
                    print("buy ", trader.trades[-1]['entry_time'], trader.trades[-1]['entry_price'])
                elif (y[0][0] < down_threshold):
                    trader.sell(size = 1)
                    take_point = 40
                    stop_point = 20
                    adjust_point = 30
                    in_price = df_5min['Close'][-1]
                    print("sell ", trader.trades[-1]['entry_time'], trader.trades[-1]['entry_price'])

            elif (self_position != 'None'):
                now_price = df_5min['Close'][-1]
                price_diff = now_price - in_price
                print("now status:", self_position, "now price:", now_price, datetime.datetime.now(), "price diff:", price_diff, "pnl:", trader.position()['pnl'])
                print("take point:", take_point, "stop point:", stop_point, "adjust point:", adjust_point)
                if time_condition3:
                    if self_position == 'Buy':
                        trader.sell(size = 1)
                        print("close 13:30\n")
                    elif self_position == 'Sell':
                        trader.buy(size = 1)
                        print("close 13:30\n")
                
                # this place can add stop or limit order
                elif self_position == 'Buy':    
                    # if price_diff <= -1 * stop_point: # stop loss
                    #     trader.sell(size = 1)
                    #     print("close stop loss\n")
                    # elif price_diff >= take_point: # take profit
                    #     trader.sell(size = 1)
                    #     print("close take profit\n")
                    if price_diff >= adjust_point:
                        stop_point = -25
                        print("Adjust stop point from 15 to -25\n")
                
                elif self_position == 'Sell':    
                    # if price_diff >= stop_point: # stop loss
                    #     trader.buy(size = 1)
                    #     print("close stop loss\n")
                    # elif price_diff <= -1 * take_point: # take profit
                    #     trader.buy(size = 1)
                    #     print("close take profit\n")
                    if price_diff <= -1 * adjust_point:
                        stop_point = -25
                        print("Adjust stop point from 15 to -25\n")


api.quote.unsubscribe(
    contract,
    quote_type=sj.constant.QuoteType.BidAsk,
    version=sj.constant.QuoteVersion.v1
)

api.logout()
