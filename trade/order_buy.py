import sys, os

import time
import datetime as dt
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from ex_upbit import api_exchange as upbit_api
from database import message as dbgout

def order(upbit, ticker):
    try:
        #cash = round(upbit.get_balance(ticker='KRW'), 2)
        #cash = upbit.get_balance(ticker='KRW')
        cash = upbit.get_balance(ticker='KRW')
        print(cash)
        chance = upbit.get_chance(ticker=ticker)
        fee = float(chance['ask_fee'])  ## 시장가 주문에 대한 수수료, 지정가 주문에 대한 수수료는 maker_ask_fee
        min_limit = float(chance['market']['ask']['min_total'])
        current = round(upbit_api.get_current_price(ticker),2)
        coins = round(cash / current, 2)
        print("cash:", cash, ",fee:", fee, ",min_limit:", min_limit, ",current:", current, ",coins:", coins)

        # 보유 현금으로 최대한 많은 coin을 매수한다.
        while True:
            amount = round((coins * current) * (1.0 + fee), 2)
            if (cash > min_limit) and (cash > amount):
                break
            else:
                coins = coins - 1.0
                if coins < 0.0:
                    message = 'Not enough money'
                    #dbgout.printf(To='telegram', From='order_buy()', data=message)
                    print(message)
                    return (False, 0.0, 0.0, 0.0)
        #print("cash:", cash, ",fee:", fee, ",min_limit:", min_limit, ",current:", current, ",coins:", coins)

        # 시장가 매수
        order_price = upbit_api.get_tick_size(coins * current)
        #resp = upbit_core.buy_market_order(ticker=todo.ticker, price=order_price)

        # 매수 주문후 정보 취합.
        fee = round(order_price * fee, 2)
        coins = round((order_price-fee)/current, 8)
        #print("order_price:", order_price, ",fee:", fee, ",coins:", coins, ",real_order:", order_price-fee)

        #dbgout.printf(To='file', From='order_buy()', data=todo)
        #dbgout.printf(To='telegram', From='order_buy()', data=todo)

        return (True, order_price, fee, coins)

    except Exception as e:
        #dbgout.printf(To='telegram', From='order_buy()', data=str(e))
        print(e)
        return (False, 0.0, 0.0, 0.0)

	#print(result)
    #return [True, result]

if __name__ == '__main__':
    access = "dcA4JFsGWHRJcqnSmGNRM3FdMCLt4J4i2dSxX7uD"
    secret = "QuxtgJxiyhxUz9xuuvq9O2qIdxxSclvQIas9bWWn"
    upbit  = upbit_api.UPBIT(access, secret)
    print(upbit)

    (ask, order, fee, coins) = order(upbit, "KRW-ELF")
    print("ASK?:", ask, ",ORDER:", order, ",FEE:", fee, ",COINS:", coins)