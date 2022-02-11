import warnings
import logging
import os
import glob
from typing import List

from multiprocessing import Pool

from pprint import pprint
from datetime import datetime, timedelta

import pandas as pd
from jqdatasdk import get_trade_days, auth, is_auth, logout
from jqdatasdk.api import get_all_securities, get_extras, get_price
from jqdatasdk.technical_analysis import *


def authenticate():
    if is_auth():
        # print(f'Is already authed: {is_auth()}')
        pass
    else:
        auth('15721652180', '60Pacmer')
        assert is_auth()


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')




def fetch_one_day_trend(date, stocks: List[str] = None):
    authenticate()
    logging.info(f'Downloading {date}')



    if os.path.exists(f'daily_data/df_{date}.csv'):
        logging.info(f'{date} is already downloaded')
        # df = pd.read_csv(f'daily_data/df_{date}.csv')
    else:

        with warnings.catch_warnings():
            warnings.simplefilter('ignore') # ignore pandas version, PanelObsoleteWarning
            if stocks is None:
                stocks = list(get_all_securities(types=['stock'], date=date).index)
                # stocks_left = [stock for stock in stocks if not stock.startswith('300') and not stock.startswith('688')]
                stocks = [stock for stock in stocks
                                if stock.startswith('00') or stock.startswith('60')]
                # non-ST
                res = get_extras('is_st', stocks, start_date=date, end_date=date, df=True)
                stocks = [stock for stock in stocks if not res[stock].item()]


            fields = ['open', 'close', 'factor', 'high_limit', 'low_limit', 'pre_close', 'avg', 'low', 'high']
            # fields = ['open', 'close', 'factor', 'high_limit', 'low_limit', 'pre_close', 'avg', 'low', 'high', 'volume', 'money']

            K, D, J = KDJ(stocks, check_date=date, N=9, M1=3, M2=3, unit='1d')
            symbols = list(K.keys())
            k1d = list(K.values())
            d1d = list(D.values())
            j1d = list(J.values())

            K, D, J = KDJ(stocks, check_date=date, N=9, M1=3, M2=3, unit='120m')
            k2h = list(K.values())
            d2h = list(D.values())
            j2h = list(J.values())


            df_kdj = pd.DataFrame.from_dict({
                'symbol': symbols,
                'k1d': k1d,
                'd1d': d1d,
                'j1d': j1d,
                'k2h': k2h,
                'd2h': d2h,
                'j2h': j2h,
                # 'rsi_6d': rsi_6d_values,
                # 'rsi_12d': rsi_12d_values,
                # 'rsi_24d': rsi_24d_values,
                # 'rsi_30m': rsi_30m_values,
                # 'rsi_60m': rsi_60m_values
            })

            # print(f'date type: {type(date), date}')
            df_kdj['date'] = date

            df = get_price(stocks, start_date=date, end_date=date, frequency='1d', skip_paused=False, fields=fields)

            df.rename(columns={'code': 'symbol'}, inplace=True)
            df['date'] = df['time'].dt.strftime('%Y-%m-%d')
            df.reset_index(drop=True, inplace=True)

            df = pd.merge(df, df_kdj, on=['date', 'symbol'], how='left')
            df.to_csv(f'daily_data/df_{date}.csv', index=False, float_format='%.6f')

    logout()

if __name__ == '__main__':
    stocks = ['600706.XSHG', '600847.XSHG', '002932.XSHE']

    from_date = '2021-06-01'
    data_dir = 'daily_data'

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    now = datetime.now()
    td = timedelta(days=0) if now.strftime("%H%M") >= '1505' else timedelta(days=1)
    to_date = (datetime.now() - td).strftime('%Y-%m-%d')

    authenticate()
    trade_days = [date.strftime('%Y-%m-%d')
                  for date in get_trade_days(start_date=from_date, end_date=to_date)]
    logout()

    dates_fetched = sorted([fname[-14:-4] for fname in glob.glob(f'{data_dir}/df_*.csv')])
    pprint(sorted(dates_fetched))
    dates_to_be_fetched = sorted(set(trade_days) - set(dates_fetched))
    pprint(sorted(dates_to_be_fetched))
    # dates_to_be_fetched = sorted(set(trade_days))

    with Pool(processes=3) as pool:
        pool.map(fetch_one_day_trend, dates_to_be_fetched)

    df_list = []
    for date in trade_days:
        df = pd.read_csv(f'daily_data/df_{date}.csv')
        df_list.append(df)
    df_final = pd.concat(df_list)
    # df = download_(, '2022-01-04')
    # print(df.shape)
    df_final.to_csv('df.csv', index=False)


    # fetch_one_day_trend('2022-02-09', stocks=stocks)




