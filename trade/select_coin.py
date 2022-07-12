# %%

import os, sys
import shutil
import glob
import datetime as dt
import time
import pandas as pd
import numpy as np
import colorama


WINS = 0.49		## 승률. 투자할 코인의 과거 경력으로 시뮬레이션했을때 승률이 이 기준을 통과한 코인만 선택
MAX_INVEST = 5	## 동시에 투자 가능한 코인의 최대 수

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from ex_upbit import api_exchange as upbit

def progress_bar(progress, total, color=colorama.Fore.YELLOW):
	percent = 100 * (progress / float(total))
	bar = '#' * int(percent) + '-' * (100 - int(percent))
	print(color + f"\r|{bar}| {percent:.2f}%", end="\r")
	if float(progress) >= float(total):
		print(colorama.Fore.GREEN + f"\r|{bar}| {percent:.2f}%", end="\r")

def prepare_database(db_dir=None, except_tickers=None):
# 원화거래 가능 종목 조회
# fiat를 'KRW'로 지정해서 원화 거래 가능한 비트코인명만 수집
# 업비트 원화 거래 가능한 티커명 조회
	tickers = upbit.get_tickers(fiat='KRW')
	ticker_names = [x for x in tickers if x not in except_tickers]
	print("download: " + str(len(ticker_names)) + "tickers")

	# 현재 시간 data base directory가 없으면 새로 생성 
	if not os.path.exists(db_dir):
		os.makedirs(db_dir)

	# 어제 download한 database는 모두 삭제한다.
	yesterday = dt.date.today() - dt.timedelta(days=1)
	rm_dir = db_dir[:len(db_dir)-19] + str(yesterday) + "*"
	rm_dirs = glob.glob(rm_dir)
	for rm_dir in rm_dirs:
		shutil.rmtree(rm_dir)

	# 현재 시간 기준 일봉 데이터 다운로드
	idx = 0
	progress_bar(0, len(ticker_names))
	for ticker in ticker_names:
		data_file = db_dir + '/' + ticker + '.csv'
		if os.path.exists(data_file) is True:
			pass
		else:
			df = upbit.get_ohlcv(ticker, interval='minutes60')
			df.to_csv(data_file, encoding='cp949')
			time.sleep(0.5)
		progress_bar(idx+1, len(ticker_names))
		idx += 1
	print(colorama.Fore.RESET)

	# 거래대금 정보 가져오기
	tickers = []
	for ticker_name in ticker_names:
		data_file = db_dir + '/' + ticker_name + '.csv'
		df = pd.read_csv(data_file)    
		yesterday = df.iloc[-2]    
		tickers.append([ticker_name, yesterday['volume'], yesterday['value']])

	#print(df)
	df = pd.DataFrame(tickers)
	df = df.rename(columns = {0:'ticker', 1:'volume', 2:'value'})

	# 거래량, 거래대금을 내림차순으로 정렬
	df = df.sort_values(by=['volume', 'value'], ascending=False, inplace=False)
	df = df.reset_index(drop=True)
	#print(df)

	# 상위 종목만 리스트업
	#df = df.loc[:9]
	df = df.loc[:29]
	#print(df)

	return df

# Coin별 정보 저장 class ======================================================
class Coin:
	def __init__(self, ticker_name):
		# 티커명
		self.ticker_name = ticker_name
		self.best_k  = 0
		self.worst_k = 0

		# 기간 적용 수익률 : 변동성 돌파 전략 
		self.hpr = 0	# 기간 수익률
		self.simple_hpr = 0 # 단순 보유 수익률
		self.mdd = 0 	    # 최대 낙폭
		self.win_count = 0  # 승
		self.lose_count = 0 # 패
		self.win_rate = 0   # 승률

	def get_winning_rate(self):
		return self.win_count / (self.win_count+self.lose_count)

	def __str__(self):
		return (f'{self.ticker_name}  '
				 ', k:'		+ f'{self.best_k:.2f}'
		 		 ', hpr:'		+ f'{self.hpr:.2f}'
		 		 ', mdd:'			+ f'{self.mdd:.2f}'
		 		 ', s_hpr:'	+ f'{self.simple_hpr:.2f}'
		 		 ', win:'             + f'{self.win_count:.0f}' 
		 		 ', lose:'             + f'{self.lose_count:.0f}'
		 		 ', win_rate:'            + f'{self.win_rate:.2f}'
				)

# 변동성 돌파 전략 함수 구현
#   티커명(코인명)을 break_out()에 전달하면 일봉 데이터 파일을 읽어서 기간수익률이
#     가장 높게 나오는 최적의 k를 찾고 Coin을 리턴
#   수익률 계산시 슬리피지와 업비트 수수료를 반영
def findout_K(dir_path, ticker_name):    
	data_file = dir_path + '/' + ticker_name + '.csv'
	#print(data_file)
	df = pd.read_csv(data_file)
	best_df = None
	#print(df)
    
	# 분석할 ticker 초기화
	coin = Coin(ticker_name)

	#print(coin)

	# 단순 보유 수익률
	coin.simple_hpr = df.iloc[-1]['close'] / df.iloc[0]['close'] 

	# 변동성 크기
	df['range'] = (df['high'] - df['low']).shift(1)
	#print(df)

	# 과거 200개의 데이터(1시간 간격)를 이용해서 K값에 때른 결과를 emulation한다.
	for k in np.arange(0.10, 0.99, 0.01):
		df['K'] = k
		# 목표가
		df['target_price'] = df['open'] + df['range'] * k
		df['buy'] = np.where((df['high'] > df['target_price']), 1, 0)

		# 슬리피지 + 업비트 매도/매수 수수료 (0.05% * 2)
		fee = 0.002 + 0.001

		# buy가 1이면, 수익률 = 종가/목표가 - 수수료
		# buy가 0이면, 수익률 = 1
		# ror = Rate of Return
		df['ror'] = np.where(df['buy'] == 1,
							 df['close'] / df['target_price'] - fee, 1)

		# hpr (Holding Period Return): 기간수익
		df['hpr'] = df['ror'].cumprod()

		# MDD (MAximum Draw Down): 최대 손실률
		# MDD = (high - low) / high * 100                
		df['dd'] = (df['hpr'].cummax() - df['hpr']) / df['hpr'].cummax() * 100

		# 승
		df['win'] = np.where(df['ror'] > 1, 1, 0)

		# 패
		df['lose'] = np.where(df['ror'] < 1, 1, 0)

		# 기간수익률
		if coin.hpr < df.iloc[-1]['hpr']: # 기간수익률이 최고일때 K값 를 찾는다.
			best_df = df.copy()            
			coin.best_k = k
			coin.worst_k = 0.99 - k
			coin.hpr = df.iloc[-1]['hpr']
			coin.mdd = df['dd'].max()
			coin.win_count = df['win'].sum()
			coin.lose_count = df['lose'].sum()
			coin.win_rate = coin.get_winning_rate()
		else:
			pass

		#print('Enter Key-in')
		#input()

	#print(df)
	#print(coin)
	return coin

# 코인 종목에 대해서 변동성 돌파전략 적용
def strategy(db_path, tickers):
	ret = []
	found_K = []
	print("analysis: " + str(len(tickers)) + "tickers")
	progress_bar(0, str(len(tickers)))
	for idx in tickers.index:
		ret = findout_K(db_path, tickers['ticker'][idx]) # 각 coin에 대해서 기간수익률이 높은 K값을 찾는다.    
		found_K.append([ret.best_k, ret.hpr, ret.mdd, ret.win_rate])
		progress_bar(idx+1, str(len(tickers)))
	print(colorama.Fore.RESET)

	df = pd.DataFrame(found_K)
	df = df.rename(columns = {0:'K', 1:'HPR', 2:'MDD', 3:'WIN_RATE'})

	tickers = pd.merge(tickers, df, left_index=True, right_index=True)
	#print(tickers)

	# 기간수익률 기준 내림차순 정렬
	tickers = tickers.sort_values(by=['HPR'], ascending=False, inplace=False)
	tickers = tickers.reset_index(drop=True)
	#print(tickers)

	# 승 + 패 의 값이 10 이하이면 데이터 분석하기에 너무 잛은 역사를 가지고 있는 코인이다. 제거.
	#tickers = tickers.loc[(tickers.win_count + tickers.lose_count) > 10]
	#tickers.reset_index(drop=True, inplace=True)
	#print(tickers)

	#승률이 50% 이상인 것만 고른다. 항상 이길수는 없다.
	invest_coins = tickers.loc[(tickers.WIN_RATE>WINS) & (tickers.WIN_RATE!=1.0)]
	invest_coins = invest_coins.reset_index(drop=True)
	invest_coins['invest'] = False
	#print(invest_coins)

	# 투자할 코인은 한번에 MAX_INVEST 개로 한정한다.
	if len(invest_coins.index) > MAX_INVEST:
		invest_coins = invest_coins.head(MAX_INVEST)

	return invest_coins

if __name__ == '__main__':
	today = dt.date.today()
	now_H = dt.datetime.now().strftime('%Y-%m-%d %H:00:00')
	now_H = dt.datetime.strptime(now_H, '%Y-%m-%d %H:%M:%S')
	print(now_H)

	## 시간 간격으로 시장에서 거래되는 코인의 데이터를 확보한다.
	except_tickers = [ 'KRW-BTC', ] # 투자에서 제외할 코인
	db_dir  = os.path.join(os.getcwd(), 'data/upbit/' + str(now_H))
	print(db_dir)
	tickers = prepare_database(db_dir, except_tickers)
	#print(tickers)

	## 변동성돌 전략으로 투자대상을 찾는다.
	invest = strategy(db_dir, tickers)
	print(invest)


# %%
