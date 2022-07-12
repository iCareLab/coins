import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import time
import datetime as dt
import pandas as pd
from database import message as dbgout
from ex_upbit import api_exchange as upbit_api

def order(upbit, todo):
    try:
        resp = upbit.get_order(todo.ticker)
        if not resp:
            pass
        else:   # 현재 주문중이면 주문을 취소한다.
            uuid = resp[0]['uuid']
            upbit.cancel_order(uuid)


        # 시장가 매도 주문
        coins = upbit.get_balance_t(todo.ticker)
        #print(coins)
        #upbit.sell_market_order(ticker=todo.ticker, volume=coins)

        # 매도 완료 후 정보 취합.
        time.sleep(1)
        todo.selled = round(upbit_api.get_current_price(todo.ticker), 2)   # Todo: 업비트의 거래내역 조회로 정확한 데이터를 찾는다.
        time.sleep(0.1)
        todo.coins = round(upbit.get_balance(ticker=todo.ticker), 2)
        time.sleep(0.1)
        todo.cash = round(upbit.get_balance(ticker='KRW'), 2)
        #print(todo)

        ## ToDo file update
        today = dt.date.today()
        todo_file = 'ToDo-' + str(today.strftime('%Y-%m'))
        today_todo = pd.read_csv(todo_file)
        today_todo = today_todo.astype({'date':'datetime64'})
        #print(today_todo)

        today_todo.selled.iloc[-1]  = todo.selled
        today_todo.coins.iloc[-1]  = todo.coins
        today_todo.cash.iloc[-1]  = todo.cash
        today_todo.to_csv(todo_file, mode='w', index=False)
        #print(today_todo)

        dbgout.printf(To='file', From='order_sell()', data=todo)
        dbgout.printf(To='telegram', From='order_sell()', data=todo)

        return False

    except Exception as e:
        dbgout.printf(to='telegram', message=str(e))
        time.sleep(1)

def backtesting(data, result):
    index = result.index[-1]
    #print(index)

    coins = result.coins.iloc[index-1]
    current = data[1].close
    selled = round(coins * current, 2)

    cash = round(result.cash.iloc[index-1] + selled, 2)

    profit = round(selled - result.buyed.iloc[index-1], 2)

    result.loc[index] = [
		data[0],						# date
		result.ticker.iloc[index-1],		# ticker
		result.best_K.iloc[index-1],		# best_K
		result.target.iloc[index-1],		# target price
		result.stop_loss.iloc[index-1],	# stop_loss price

        current,    # unit price(current)
		0.0,		# buy price
		selled,		# sell price
		0.0,		# coin count
		cash,		# cash amount

		profit,							# daily profit
		result.monthly.iloc[index-1],		# monthly profit
		result.yearly.iloc[index-1]		# yearly profit
	]


    print( f'BT[{str(data[0])}]' 
		 + f' {result.ticker.iloc[index]} ' 
		 + f' BEST_K:{result.best_K.iloc[index]:,}'
		 + f' TARGET:{result.target.iloc[index]:,}'
		 + f' SELL:{selled:,}'
		 + f'\tPRICE:{current:,}'
		 + f' CASH:{cash:,}'
		 + f' PROFIT:{profit:,}'
		)


    result.loc[index+1] = result.loc[index]

	#print(result)
    return [False, result]