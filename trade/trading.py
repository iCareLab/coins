
'''
* 장 종료시간에서 모든 코인을 처리하고 데이터를 정리한 이후 새로운 장시작을 개시한다. 이때
  모든 새로운 데이터를 새로 받아 재 계산을 한다. 즉, 24시간 단위의 새로운 데이터 취합을
  1시간 단위의 데이터 취합으로 처리한다.
'''
import os, sys
import platform
import time
import pandas as pd
import numpy as np
import datetime as dt
import schedule
import colorama

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from database import message as dbgout
from ex_upbit import api_exchange as upbit_api
from trade import select_coin as coin
from trade import order_buy as buy
from trade import order_sell as sell

MAX_CASH = 100000 # 최대 투자 금액 (10만원)

def get_OHLCV_data(token, _date_time, interval='minute3'):
	# target일 데이터를 가져온다.
    if platform.processor() == 'x86_64':
        #df = upbit_api.get_ohlcv(token, count=520, interval='minute3', to=str(_date_time))
        df = upbit_api.get_ohlcv(token, count=520, interval=interval, to=str(_date_time))
        df = df.drop_duplicates()
    else:
        #df = upbit_api.get_ohlcv(token, count=480, interval='minute3', to=str(_date_time))
        df = upbit_api.get_ohlcv(token, count=480, interval=interval, to=str(_date_time))

    #print(df)
    return df

def target_price(invest, now_hour):
    ## 1. 전날의 일봉 기준 range(= 전일 고가 - 전일 저가)를 계산합니다.
    ## 2. 당일 장중 가격이 당일시가 + (전일 range 값 * K)을 넘을 경우 매수 합니다. (K = 노이즈비율)
    ## 3. 익일 시가 기준으로 지정가 매도를 합니다.
    ##    price_buy = (open + (prev_high - perv_low) * best_K) * (1+FEE)
    print("Target Price: " + str(len(invest)) + "tickers")
    price = []
    coin.progress_bar(0, len(invest))
    for idx in invest.index:
        df = get_OHLCV_data(invest.iloc[idx].ticker, now_hour) # 3분간격의 데이터를 가져온다.
        highst_price = int(np.max(df.close))	# 24시간 동안의 데이터중에 최고가
        lowest_price  = int(np.min(df.close))	# 24시간 동안의 데이터중에 최저가
        current_price = upbit_api.get_current_price(invest.iloc[idx].ticker)

        '''변동성 돌파 전략으로 매수 목표가 계산'''
        target_price = round(current_price + (highst_price - lowest_price) * invest.iloc[idx].K, 2)
        break_price = round(target_price * 0.9, 2)
        price.append([current_price, target_price, break_price])
        coin.progress_bar(idx+1, len(invest))
    print(colorama.Fore.RESET)

    df_price = pd.DataFrame(price)
    df_price = df_price.rename(columns = {0:'price', 1:'target', 2:'break'})

    invest = pd.merge(invest, df_price, left_index=True, right_index=True)
    #print(invest)

    return invest

def market_state(invest):
    try:
        print("Market Status Analysis")
        coin.progress_bar(0, len(invest))
        for idx in invest.index:
            df = get_OHLCV_data(invest.iloc[idx].ticker, now_hour, 'minutes240') # 4시간 간격의 데이터를 가져온다.
            #print(df)
            ma5 = df['close'].rolling(window=5).mean()
            last_ma5 = ma5[-2]

            price = upbit_api.get_current_price(invest.iloc[idx].ticker)
            if price > last_ma5:
                invest['market'] = "increase"
                #print("상승장")
            else:
                invest['market'] = "decrease"
                #print("하락장")

            coin.progress_bar(idx+1, len(invest))
        print(colorama.Fore.RESET)

    except Exception as e:
        dbgout.printf(to='telegram', message=str(e))
        time.sleep(1)

def date_range(start_date, end_date):
	for n in range(int((end_date - start_date).days)):
		yield start_date + dt.timedelta(n)

# 자동매매 시작 #######################################################
def auto_trading(market, invest):
    try:
        pass

    except Exception as e:
        print(e)
        time.sleep(1)

if __name__ == '__main__':
    access = "dcA4JFsGWHRJcqnSmGNRM3FdMCLt4J4i2dSxX7uD"
    secret = "QuxtgJxiyhxUz9xuuvq9O2qIdxxSclvQIas9bWWn"
    UPBIT  = upbit_api.UPBIT(access, secret)

    today = dt.date.today()
    now_hour = dt.datetime.now().strftime('%Y-%m-%d %H:00:00')
    now_hour = dt.datetime.strptime(now_hour, '%Y-%m-%d %H:%M:%S')
    print(now_hour)

	## 시간 간격으로 시장에서 거래되는 코인의 데이터를 확보한다.
    except_tickers = [ 'KRW-BTC', ] # 투자에서 제외할 코인
    dir_path = os.path.join(os.getcwd(), 'data/upbit/' + str(now_hour))
    tickers = coin.prepare_database(dir_path, except_tickers)
	#print(tickers)

	## 변동성돌파 전략으로 투자대상을 찾는다.
    invest = coin.strategy(dir_path, tickers)
    #print(invest)

    invest = target_price(invest, now_hour)
    #print(invest)

    market_state(invest)
    print(invest)

    ask = bid = False
    order = fee = coins = 0.0
    for idx in invest.index:
        if (invest.iloc[idx].invest == False) \
           and (invest.iloc[idx].market == 'increase') \
           and (invest.iloc[idx].target >= 1.0):    ## target 매수금액이 1원 이상이여야 투자대상으로 선택한다.
            print(invest.iloc[idx].ticker, ": 투자 하자")
            price = upbit_api.get_current_price(invest.iloc[idx].ticker)
            if(price > invest.iloc[idx].target):
                (ask, order, fee, coins) = buy.order(UPBIT, invest.iloc[idx].ticker)

        elif (invest.iloc[idx].invest == True):
            print(invest.iloc[idx].ticker, ": 회수 하자")
            selled = sell.order(UPBIT, invest.iloc[idx].ticker)

        else:
            print(invest.iloc[idx].ticker, ": 투자하지 말자")

        # invest table 내용 업데이트.
        if ask is True:
            print("ASK?:", ask, ",ORDER:", order, ",FEE:", fee, ",COINS:", coins)
            ask = False
            pass

        if bid is True:
            print("BID?:", ask, ",ORDER:", order, ",FEE:", fee, ",COINS:", coins)
            pass
