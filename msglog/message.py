import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import datetime as dt
import telegram as tel
from ex_upbit import api_exchange as upbit_api

## 텔레그램 BOT의 token과 chat_id
bot = tel.Bot(token="5021511935:AAGcz91cnamedpZcE2BIePXXokc8Zd-cPaQ")
chat_id = 5079868308

def save_to_file(text):
    today = dt.date.today()
    log_path = os.path.join(os.getcwd(), './log/' )

    if not os.path.exists(log_path):		# directory가 없으면 만들기
        os.makedirs(log_path)

    this_month = str(today.year) + '-' + str(today.month)
    filename = log_path + this_month + '_invest.log'
    if os.path.exists(filename):
        fp = open(filename, 'a')
    else:
        print(filename)
        fp = open(filename, 'w')
    
    fp.write(text + '\n')
    fp.close()

def printf(To=None, From=None, data=None):
    message = dt.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

    if From == 'get_todo()':
        current = upbit_api.get_current_price(str(data.ticker))
        if To == 'file':
            message += f'-{From}-'
            message += ( f' TOKEN:{data.ticker}' 
                    +f', BEST_K:{data.best_K}' 
                    +f', TARGET:{data.target:,}'
                    +f', CURRENT:{current:,}' )

        if To == 'telegram':
            message += ( f'\nTOKEN: {data.ticker}' 
                    +f'\nBEST_K: {data.best_K}' 
                    +f'\nTARGET: {data.target:,}'
                    +f'\nCURRENT: {current:,}' )
    
    if From == 'daily_closing()':
        if To == 'file':
            message += ( f' 현금(KRW):{data.cash:,}' )
        if To == 'telegram':
            message += ( f'\n현금(KRW): {data.cash:,}' )

    if From == 'order_buy()':
        if type(data) is str:
            message += ('\n' + data)
        else:
            if To == 'file':
                message += f'-{From}-'
                message += ( f' {data.ticker}'
                        + ', BUY:{:0,.2f}'.format(data.buyed)
                        + ', COIN:{:0,.2f}'.format(data.coins)
                        + ', CASH:{:0,.2f}'.format(data.cash)
                )
            if To == 'telegram':
                message += ( f'\n{data.ticker}'
                        + '\nBUY: {:0,.2f}'.format(data.buyed)
                        + '\nCOIN: {:0,.2f}'.format(data.coins)
                        + '\nCASH: {:0,.2f}'.format(data.cash)
                )

    if From == 'order_sell()':
        if type(data) is str:
            message += ('\n' + data)
        else:
            if To == 'file':
                message += f'-{From}-'
                message += ( f' {data.ticker}'
                        + ', SELL:{:0,.2f}'.format(data.selled)
                        + ', CASH:{:0,.2f}'.format(data.cash)
                )
            if To == 'telegram':
                message += ( f'\n{data.ticker}'
                        + '\nSELL: {:0,.2f}'.format(data.selled)
                        + '\nCASH: {:0,.2f}'.format(data.cash)
                )

    if To == 'telegram':
        #bot.sendMessage(chat_id=chat_id, text=message)   # 텔레그램으로 메세지 보내기
        pass
    elif To == 'file':
        save_to_file(text=message)
        print(message)

