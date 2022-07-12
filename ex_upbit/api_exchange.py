# -*- coding: utf-8 -*-

"""
pyupbit.exchange_api

This module provides exchange api of the Upbit API.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import math
import jwt          # PyJWT
import re
import uuid
import hashlib
import time
import datetime as dt
import pandas as pd
from urllib.parse import urlencode
#from api_request import _send_get_request, _send_post_request, _send_delete_request, _call_public_api
from ex_upbit import api_request as req

def get_tick_size(price, method="floor"):
    """원화마켓 주문 가격 단위 

    Args:
        price (float): 주문 가격 
        method (str, optional): 주문 가격 계산 방식. Defaults to "floor".

    Returns:
        float: 업비트 원화 마켓 주문 가격 단위로 조정된 가격 
    """

    if method == "floor":
        func = math.floor
    elif method == "round":
        func = round 
    else:
        func = math.ceil 

    if price >= 2000000:
        tick_size = func(price / 1000) * 1000
    elif price >= 1000000:
        tick_size = func(price / 500) * 500
    elif price >= 500000:
        tick_size = func(price / 100) * 100
    elif price >= 100000:
        tick_size = func(price / 50) * 50
    elif price >= 10000:
        tick_size = func(price / 10) * 10
    elif price >= 1000:
        tick_size = func(price / 5) * 5
    elif price >= 100:
        tick_size = func(price / 1) * 1
    elif price >= 10:
        tick_size = func(price / 0.1) / 10
    else:
        tick_size = func(price / 0.01) / 100

    return tick_size


class UPBIT:
    def __init__(self, access, secret):
        self.access = access
        self.secret = secret


    def _request_headers(self, query=None):
        payload = {
            "access_key": self.access,
            "nonce": str(uuid.uuid4())
        }

        if query is not None:
            m = hashlib.sha512()
            m.update(urlencode(query, doseq=True).replace("%5B%5D=", "[]=").encode())
            query_hash = m.hexdigest()
            payload['query_hash'] = query_hash
            payload['query_hash_alg'] = "SHA512"

        #jwt_token = jwt.encode(payload, self.secret, algorithm="HS256").decode('utf-8')
        jwt_token = jwt.encode(payload, self.secret, algorithm="HS256")     # PyJWT >= 2.0
        authorization_token = 'Bearer {}'.format(jwt_token)
        headers = {"Authorization": authorization_token}
        return headers


    #--------------------------------------------------------------------------
    # 자산 
    #--------------------------------------------------------------------------
    #     전체 계좌 조회
    def get_balances(self, contain_req=False):
        """
        전체 계좌 조회
        :param contain_req: Remaining-Req 포함여부
        :return: 내가 보유한 자산 리스트
        [contain_req == True 일 경우 Remaining-Req가 포함]
        """
        url = "https://api.upbit.com/v1/accounts"
        headers = self._request_headers()
        result = req._send_get_request(url, headers=headers)
        if contain_req:
            return result
        else:
            return result[0]


    def get_balance(self, ticker="KRW", contain_req=False):
        """
        특정 코인/원화의 잔고를 조회하는 메소드
        :param ticker: 화폐를 의미하는 영문 대문자 코드
        :param contain_req: Remaining-Req 포함여부
        :return: 주문가능 금액/수량 (주문 중 묶여있는 금액/수량 제외)
        [contain_req == True 일 경우 Remaining-Req가 포함]
        """
        try:
            # fiat-ticker
            # KRW-BTC
            if '-' in ticker:
                ticker = ticker.split('-')[1]

            balances, req = self.get_balances(contain_req=True)

            # search the current currency
            balance = 0
            for x in balances:
                if x['currency'] == ticker:
                    balance = float(x['balance'])
                    break

            if contain_req:
                return balance, req
            else:
                return balance
        except Exception as x:
            print(x.__class__.__name__)
            return None

    def get_balance_t(self, ticker='KRW', contain_req=False):
        """
        특정 코인/원화의 잔고 조회(balance + locked)
        :param ticker: 화폐를 의미하는 영문 대문자 코드
        :param contain_req: Remaining-Req 포함여부
        :return: 주문가능 금액/수량 (주문 중 묶여있는 금액/수량 포함)
        [contain_req == True 일 경우 Remaining-Req가 포함]
        """
        try:
            # KRW-BTC
            if '-' in ticker:
                ticker = ticker.split('-')[1]

            balances, req = self.get_balances(contain_req=True)

            balance = 0
            locked = 0
            for x in balances:
                if x['currency'] == ticker:
                    balance = float(x['balance'])
                    locked = float(x['locked'])
                    break

            if contain_req:
                return balance + locked, req
            else:
                return balance + locked
        except Exception as x:
            print(x.__class__.__name__)
            return None

    def get_avg_buy_price(self, ticker='KRW', contain_req=False):
        """
        특정 코인/원화의 매수평균가 조회
        :param ticker: 화폐를 의미하는 영문 대문자 코드
        :param contain_req: Remaining-Req 포함여부
        :return: 매수평균가
        [contain_req == True 일 경우 Remaining-Req가 포함]
        """
        try:
            # KRW-BTC
            if '-' in ticker:
                ticker = ticker.split('-')[1]

            balances, req = self.get_balances(contain_req=True)

            avg_buy_price = 0
            for x in balances:
                if x['currency'] == ticker:
                    avg_buy_price = float(x['avg_buy_price'])
                    break
            if contain_req:
                return avg_buy_price, req
            else:
                return avg_buy_price

        except Exception as x:
            print(x.__class__.__name__)
            return None

    def get_amount(self, ticker, contain_req=False):
        """
        특정 코인/원화의 매수금액 조회
        :param ticker: 화폐를 의미하는 영문 대문자 코드 (ALL 입력시 총 매수금액 조회)
        :param contain_req: Remaining-Req 포함여부
        :return: 매수금액
        [contain_req == True 일 경우 Remaining-Req가 포함]
        """
        try:
            # KRW-BTC
            if '-' in ticker:
                ticker = ticker.split('-')[1]

            balances, req = self.get_balances(contain_req=True)

            amount = 0
            for x in balances:
                if x['currency'] == 'KRW':
                    continue

                avg_buy_price = float(x['avg_buy_price'])
                balance = float(x['balance'])
                locked = float(x['locked'])

                if ticker == 'ALL':
                    amount += avg_buy_price * (balance + locked)
                elif x['currency'] == ticker:
                    amount = avg_buy_price * (balance + locked)
                    break
            if contain_req:
                return amount, req
            else:
                return amount
        except Exception as x:
            print(x.__class__.__name__)
            return None

    # endregion balance


    #--------------------------------------------------------------------------
    # 주문 
    #--------------------------------------------------------------------------
    #     주문 가능 정보
    def get_chance(self, ticker, contain_req=False):
        """
        마켓별 주문 가능 정보를 확인.
        :param ticker:
        :param contain_req: Remaining-Req 포함여부
        :return: 마켓별 주문 가능 정보를 확인
        [contain_req == True 일 경우 Remaining-Req가 포함]
        """
        try:
            url = "https://api.upbit.com/v1/orders/chance"
            data = {"market": ticker}
            headers = self._request_headers(data)
            result = req._send_get_request(url, headers=headers, data=data)
            if contain_req:
                return result
            else:
                return result[0]
        except Exception as x:
            print(x.__class__.__name__)
            return None
    

    #    개별 주문 조회 
    def get_order(self, ticker_or_uuid, state='wait', kind='normal', contain_req=False):
        """
        주문 리스트 조회
        :param ticker: market
        :param state: 주문 상태(wait, done, cancel)
        :param kind: 주문 유형(normal, watch)
        :param contain_req: Remaining-Req 포함여부
        :return:
        """
        # TODO : states, identifiers 관련 기능 추가 필요
        try:
            p = re.compile(r"^\w+-\w+-\w+-\w+-\w+$")
            # 정확히는 입력을 대문자로 변환 후 다음 정규식을 적용해야 함
            # - r"^[0-9A-F]{8}-[0-9A-F]{4}-4[0-9A-F]{3}-[89AB][0-9A-F]{3}-[0-9A-F]{12}$"
            is_uuid = len(p.findall(ticker_or_uuid)) > 0
            if is_uuid:
                url = "https://api.upbit.com/v1/order"
                data = {'uuid': ticker_or_uuid}
                headers = self._request_headers(data)
                result = req._send_get_request(url, headers=headers, data=data)
            else :

                url = "https://api.upbit.com/v1/orders"
                data = {'market': ticker_or_uuid,
                        'state': state,
                        'kind': kind,
                        'order_by': 'desc'
                        }
                headers = self._request_headers(data)
                result = req._send_get_request(url, headers=headers, data=data)

            if contain_req:
                return result
            else:
                return result[0]
        except Exception as x:
            print(x.__class__.__name__)
            return None


    def get_individual_order(self, uuid, contain_req=False):
        """
        주문 리스트 조회
        :param uuid: 주문 id
        :param contain_req: Remaining-Req 포함여부
        :return:
        """
        # TODO : states, uuids, identifiers 관련 기능 추가 필요
        try:
            url = "https://api.upbit.com/v1/order"
            data = {'uuid': uuid}
            headers = self._request_headers(data)
            result = req._send_get_request(url, headers=headers, data=data)
            if contain_req:
                return result
            else:
                return result[0]
        except Exception as x:
            print(x.__class__.__name__)
            return None

    #    주문 취소 접수
    def cancel_order(self, uuid, contain_req=False):
        """
        주문 취소
        :param uuid: 주문 함수의 리턴 값중 uuid
        :param contain_req: Remaining-Req 포함여부
        :return:
        """
        try:
            url = "https://api.upbit.com/v1/order"
            data = {"uuid": uuid}
            headers = self._request_headers(data)
            result = req._send_delete_request(url, headers=headers, data=data)
            if contain_req:
                return result
            else:
                return result[0]
        except Exception as x:
            print(x.__class__.__name__)
            return None


    #     주문 
    def buy_limit_order(self, ticker, price, volume, contain_req=False):
        """
        지정가 매수
        :param ticker: 마켓 티커
        :param price: 주문 가격
        :param volume: 주문 수량
        :param contain_req: Remaining-Req 포함여부
        :return:
        """
        try:
            url = "https://api.upbit.com/v1/orders"
            data = {"market": ticker,
                    "side": "bid",
                    "volume": str(volume),
                    "price": str(price),
                    "ord_type": "limit"}
            headers = self._request_headers(data)
            result = req._send_post_request(url, headers=headers, data=data)
            if contain_req:
                return result
            else:
                return result[0]
        except Exception as x:
            print(x.__class__.__name__)
            return None

    def buy_market_order(self, ticker, price, contain_req=False):
        """
        시장가 매수
        :param ticker: ticker for cryptocurrency
        :param price: KRW
        :param contain_req: Remaining-Req 포함여부
        :return:
        """
        try:
            url = "https://api.upbit.com/v1/orders"
            data = {"market": ticker,  # market ID
                    "side": "bid",  # buy
                    "price": str(price),
                    "ord_type": "price"}
            headers = self._request_headers(data)
            result = req._send_post_request(url, headers=headers, data=data)
            if contain_req:
                return result
            else:
                return result[0]
        except Exception as x:
            print(x.__class__.__name__)
            return None

    def sell_market_order(self, ticker, volume, contain_req=False):
        """
        시장가 매도 메서드
        :param ticker: 가상화폐 티커
        :param volume: 수량
        :param contain_req: Remaining-Req 포함여부
        :return:
        """
        try:
            url = "https://api.upbit.com/v1/orders"
            data = {"market": ticker,  # ticker
                    "side": "ask",  # sell
                    "volume": str(volume),
                    "ord_type": "market"}
            headers = self._request_headers(data)
            result = req._send_post_request(url, headers=headers, data=data)
            if contain_req:
                return result
            else:
                return result[0]
        except Exception as x:
            print(x.__class__.__name__)
            return None

    def sell_limit_order(self, ticker, price, volume, contain_req=False):
        """
        지정가 매도
        :param ticker: 마켓 티커
        :param price: 주문 가격
        :param volume: 주문 수량
        :param contain_req: Remaining-Req 포함여부
        :return:
        """
        try:
            url = "https://api.upbit.com/v1/orders"
            data = {"market": ticker,
                    "side": "ask",
                    "volume": str(volume),
                    "price": str(price),
                    "ord_type": "limit"}
            headers = self._request_headers(data)
            result = req._send_post_request(url, headers=headers, data=data)
            if contain_req:
                return result
            else:
                return result[0]
        except Exception as x:
            print(x.__class__.__name__)
            return None


    #--------------------------------------------------------------------------
    # 출금
    #--------------------------------------------------------------------------
    #     개별 출금 조회
    def get_individual_withdraw_order(self, uuid: str, currency: str, contain_req=False):
        """
        현금 출금
        :param uuid: 출금 UUID
        :param txid: 출금 TXID
        :param currency: Currency 코드
        :param contain_req: Remaining-Req 포함여부
        :return:
        """
        try:
            url = "https://api.upbit.com/v1/withdraw"
            data = {"uuid": uuid, "currency": currency}
            headers = self._request_headers(data)
            result = req._send_get_request(url, headers=headers, data=data)
            if contain_req:
                return result
            else:
                return result[0]
        except Exception as x:
            print(x.__class__.__name__)
            return None


    #     코인 출금하기  
    def withdraw_coin(self, currency, amount, address, secondary_address='None', transaction_type='default', contain_req=False):
        """
        코인 출금
        :param currency: Currency symbol
        :param amount: 주문 가격
        :param address: 출금 지갑 주소
        :param secondary_address: 2차 출금주소 (필요한 코인에 한해서)
        :param transaction_type: 출금 유형
        :param contain_req: Remaining-Req 포함여부
        :return:
        """
        try:
            url = "https://api.upbit.com/v1/withdraws/coin"
            data = {"currency": currency,
                    "amount": amount,
                    "address": address,
                    "secondary_address": secondary_address,
                    "transaction_type": transaction_type}
            headers = self._request_headers(data)
            result = req._send_post_request(url, headers=headers, data=data)
            if contain_req:
                return result
            else:
                return result[0]
        except Exception as x:
            print(x.__class__.__name__)
            return None


    #     원화 출금하기
    def withdraw_cash(self, amount: str, contain_req=False):
        """
        현금 출금
        :param amount: 출금 액수
        :param contain_req: Remaining-Req 포함여부
        :return:
        """
        try:
            url = "https://api.upbit.com/v1/withdraws/krw"
            data = {"amount": amount}
            headers = self._request_headers(data)
            result = req._send_post_request(url, headers=headers, data=data)
            if contain_req:
                return result
            else:
                return result[0]
        except Exception as x:
            print(x.__class__.__name__)
            return None


    #--------------------------------------------------------------------------
    # 입금 
    #--------------------------------------------------------------------------
    #     입금 리스트 조회 
    #     개별 입금 조회
    #     입금 주소 생성 요청 
    #     전체 입금 주소 조회
    #     개별 입금 주소 조회
    #     원화 입금하기


    #--------------------------------------------------------------------------
    # 서비스 정보 
    #--------------------------------------------------------------------------
    #     입출금 현황 
    def get_deposit_withdraw_status(self, contain_req=False):
        url = "https://api.upbit.com/v1/status/wallet"
        headers = self._request_headers()
        result = req._send_get_request(url, headers=headers)
        if contain_req:
            return result
        else:
            return result[0]


    #     API키 리스트 조회
    def get_api_key_list(self, contain_req=False):
        url = "https://api.upbit.com/v1/api_keys"
        headers = self._request_headers()
        result = req._send_get_request(url, headers=headers)
        if contain_req:
            return result
        else:
            return result[0]


#--------------------------------------------------------------------------
# 티커 조회
#--------------------------------------------------------------------------
def get_url_ohlcv(interval):
    """ohlcv 요청을 위한 url을 리턴하는 함수 
    Args:
        interval (str): "day", "minute1", "minute3", "minute5", "week", "month"
    Returns:
        str: upbit api url 
    """

    if interval in ["day", "days"]:
        url = "https://api.upbit.com/v1/candles/days"
    elif interval in ["minute1", "minutes1"]:
        url = "https://api.upbit.com/v1/candles/minutes/1"
    elif interval in ["minute3", "minutes3"]:
        url = "https://api.upbit.com/v1/candles/minutes/3"
    elif interval in ["minute5", "minutes5"]:
        url = "https://api.upbit.com/v1/candles/minutes/5"
    elif interval in ["minute10", "minutes10"]:
        url = "https://api.upbit.com/v1/candles/minutes/10"
    elif interval in ["minute15", "minutes15"]:
        url = "https://api.upbit.com/v1/candles/minutes/15"
    elif interval in ["minute30", "minutes30"]:
        url = "https://api.upbit.com/v1/candles/minutes/30"
    elif interval in ["minute60", "minutes60"]:
        url = "https://api.upbit.com/v1/candles/minutes/60"
    elif interval in ["minute240", "minutes240"]:
        url = "https://api.upbit.com/v1/candles/minutes/240"
    elif interval in ["week",  "weeks"]:
        url = "https://api.upbit.com/v1/candles/weeks"
    elif interval in ["month", "months"]:
        url = "https://api.upbit.com/v1/candles/months"
    else:
        url = "https://api.upbit.com/v1/candles/days"

    return url

def get_tickers(fiat="", is_details=False, limit_info=False, verbose=False):
    """업비트 티커 조회
    Args:
        fiat (str, optional): Fiat (KRW, BTC, USDT). Defaults to empty string.
        limit_info (bool, optional): True: 요청 수 제한 정보 리턴, False: 요청 수 제한 정보 리턴 받지 않음. Defaults to False.
    Returns:
        tuple/list: limit_info가 True이면 튜플, False이면 리스트 객체  
    """
    url = "https://api.upbit.com/v1/market/all"
    detail = "true" if is_details else "false"
    markets, req_limit_info = req._call_public_api(url, isDetails=detail)

    if verbose:
        tickers = [x for x in markets if x['market'].startswith(fiat)]
    else:
        tickers = [x['market'] for x in markets if x['market'].startswith(fiat)]

    if limit_info:
        return tickers, req_limit_info
    else:
        return tickers

def get_ohlcv(ticker="KRW-BTC", interval="day", count=200, to=None, period=0.1):
    MAX_CALL_COUNT = 200
    try:
        url = get_url_ohlcv(interval=interval)

        if to == None:
            to = dt.datetime.now()
        elif isinstance(to, str):
            to = pd.to_datetime(to).to_pydatetime()
        elif isinstance(to, pd._libs.tslibs.timestamps.Timestamp):
            to = to.to_pydatetime()

        to = to.astimezone(dt.timezone.utc)

        dfs = []
        count = max(count, 1)
        for pos in range(count, 0, -200):
            query_count = min(MAX_CALL_COUNT, pos)

            to = to.strftime("%Y-%m-%d %H:%M:%S")

            contents, req_limit_info = req._call_public_api(url, market=ticker, count=query_count, to=to)
            dt_list = [dt.datetime.strptime(x['candle_date_time_kst'], "%Y-%m-%dT%H:%M:%S") for x in contents]
            df = pd.DataFrame(contents, 
                              columns=[
                                  'opening_price', 
                                  'high_price', 
                                  'low_price', 
                                  'trade_price',
                                  'candle_acc_trade_volume', 
                                  'candle_acc_trade_price'],
                              index=dt_list)
            df = df.sort_index()
            if df.shape[0] == 0:
                break
            dfs += [df]

            to = dt.datetime.strptime(contents[-1]['candle_date_time_utc'], "%Y-%m-%dT%H:%M:%S")

            if pos > 200:
                time.sleep(period)

        df = pd.concat(dfs).sort_index()
        df = df.rename(columns={"opening_price": "open", 
                                "high_price": "high", 
                                "low_price": "low", 
                                "trade_price": "close",
                                "candle_acc_trade_volume": "volume", 
                                "candle_acc_trade_price": "value"})
        return df
    except Exception as x:
        return None

def get_current_price(ticker="KRW-BTC", limit_info=False, verbose=False):
    """현재가 정보 조회
    Args:
        ticker (str/list, optional): 단일 티커 또는 티커 리스트 Defaults to "KRW-BTC".
        limit_info (bool, optional): True: 요청 제한 정보 리턴. Defaults to False.
        verbose (bool, optional): True: 원본 API 파라미터 리턴. Defaults to False.
    Returns:
        [type]: [description]
    """
    url = "https://api.upbit.com/v1/ticker"
    data, req_limit_info = req._call_public_api(url, markets=ticker)

    if isinstance(ticker, str) or (isinstance(ticker, list) and len(ticker)==1):
        # 단일 티커 
        if verbose is False:
            price = data[0]['trade_price']
        else:
            price = data[0]
    else:
        # 여러 티커로 조회한 경우 
        if verbose is False:
            price = {x['market']: x['trade_price'] for x in data}
        else:
            price = data

    if limit_info:
        return price, req_limit_info
    else:
        return price











if __name__ == "__main__":
    import pprint

    #-------------------------------------------------------------------------
    # api key
    #-------------------------------------------------------------------------
    #with open("upbit.txt") as f:
    #    lines = f.readlines()
    #    access = lines[0].strip()
    #    secret = lines[1].strip()

    access = "dcA4JFsGWHRJcqnSmGNRM3FdMCLt4J4i2dSxX7uD"
    secret = "QuxtgJxiyhxUz9xuuvq9O2qIdxxSclvQIas9bWWn"
    upbit = UPBIT(access, secret)


    #-------------------------------------------------------------------------
    # 자산 
    #     전체 계좌 조회 
    balance = upbit.get_balances()
    pprint.pprint(balance)

    #balances = upbit.get_order("KRW-XRP")
    #pprint.pprint(balances)

    # order = upbit.get_order('50e184b3-9b4f-4bb0-9c03-30318e3ff10a')
    # print(order)
    # # 원화 잔고 조회
    print(upbit.get_balance(ticker="KRW"))          # 보유 KRW
    # print(upbit.get_amount('ALL'))                  # 총매수금액
    # print(upbit.get_balance(ticker="KRW-BTC"))      # 비트코인 보유수량
    # print(upbit.get_balance(ticker="KRW-XRP"))      # 리플 보유수량

    #-------------------------------------------------------------------------
    # 주문
    #     주문 가능 정보 
    #pprint.pprint(upbit.get_chance('KRW-BTC'))

    #     개별 주문 조회
    #print(upbit.get_order('KRW-GRS'))

    # 매도
    # print(upbit.sell_limit_order("KRW-XRP", 1000, 20))

    # 매수
    # print(upbit.buy_limit_order("KRW-XRP", 200, 20))

    # 주문 취소
    # print(upbit.cancel_order('82e211da-21f6-4355-9d76-83e7248e2c0c'))

    # 시장가 주문 테스트
    # upbit.buy_market_order("KRW-XRP", 10000)

    # 시장가 매도 테스트
    # upbit.sell_market_order("KRW-XRP", 36)


    #-------------------------------------------------------------------------
    # 서비스 정보
    #     입출금 현황
    #resp = upbit.get_deposit_withdraw_status()
    #pprint.pprint(resp)

    #     API키 리스트 조회
    resp = upbit.get_api_key_list()
    print(resp)