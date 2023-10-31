import pandas as pd

def slice_time(df, start, end):
    df = df.loc[start:end]
    return df

def date_filter(df, date):
    df = df.loc[date]
    return df

def resample(df, freq='D'):
    resampled = df.resample(freq).count()
    resampled = resampled.rename(columns={resampled.columns[0]: 'count'})
    resampled = resampled['count']
    return resampled