import numpy as np
import pandas as pd
import jpholiday


def datetime_to_ymdhms(data, datetime_col, newcol_suffix=None):
    datetime_ser = data[datetime_col]
    data["year{}".format("_"+newcol_suffix if not newcol_suffix==None else "")] = pd.Index(datetime_ser).year
    data["month{}".format("_"+newcol_suffix if not newcol_suffix==None else "")] = pd.Index(datetime_ser).month
    data["day{}".format("_"+newcol_suffix if not newcol_suffix==None else "")] = pd.Index(datetime_ser).day
    data["hour{}".format("_"+newcol_suffix if not newcol_suffix==None else "")] = pd.Index(datetime_ser).hour
    data["minute{}".format("_"+newcol_suffix if not newcol_suffix==None else "")] = pd.Index(datetime_ser).minute
    data["second{}".format("_"+newcol_suffix if not newcol_suffix==None else "")] = pd.Index(datetime_ser).second
    return data
    

def make_jpholiday(date_time):
    date_time = pd.to_datetime(date_time, format='%Y%m%d').dt.date
    return date_time.map(jpholiday.is_holiday).astype(int)
    
    
def sincos_date_hour():  # https://qiita.com/shimopino/items/4ef78aa589e43f315113
    def encode(df, col):
        # この方法だと場合によって最大値が変化するデータでは正確な値は出ない
        # 例：月の日数が30日や31日の場合がある
        df[col + '_cos'] = np.cos(2 * np.pi * df[col] / df[col].max())
        df[col + '_sin'] = np.sin(2 * np.pi * df[col] / df[col].max())
        return df
    
    dates = pd.date_range(start='2017-01-01', end='2017-12-31', freq='H')
    df = pd.DataFrame(data=np.zeros(len(dates)), index=dates, columns=['test'])
    
    df['year']  = df.index.year
    df['month'] = df.index.month
    df['day']   = df.index.day
    df['hour']  = df.index.hour
    df['dow']   = df.index.dayofweek
    
    df = encode(df, 'dow')
    df = encode(df, 'hour')
    df = encode(df, 'day')
    df = encode(df, 'month')
    
    df.drop(["test"], axis=1, inplace=True)
    df.drop(df.columns[:5], axis=1, inplace=True)
    
    return df
