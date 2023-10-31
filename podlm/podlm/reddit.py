import pandas as pd
import urllib3
urllib3.disable_warnings()
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

def es_query_reddit(search: str, es: Elasticsearch):
    results = {}    
    query = {'query': {'match': {'subreddit': search}}}
    reddit_indexes = ['reddit-index-s', 'reddit-index-c']
    for ri in reddit_indexes: 
        search_results = scan(es, query = query, index=ri)
        df = es_results_to_df(search_results)
        dropcols = ['_source', '_score', '_id', '_index', 'gildings']
        df.drop(columns=dropcols, inplace=True)
        
        df = set_dtypes(df)
        df = process_datetimes(df)
        
        if ri == 'reddit-index-s':
            results['subs'] = df
            df['id'] = 't3_' + df['id']
        elif ri == 'reddit-index-c':
            results['coms'] = df    
            df['id'] = 't1_' + df['id']
    return results['subs'], results['coms']


def es_results_to_df(search_results):
    df = pd.DataFrame.from_records(list(search_results))
    df = pd.concat([df, df['_source'].apply(pd.Series)], axis=1)
    return df


def set_dtypes(df: pd.DataFrame):
    df['subreddit_type'] = df['subreddit_type'].astype('category')
    df['gilded'] = df['gilded'].astype('bool')
    return df


def process_datetimes(df: pd.DataFrame):
    df['datetime'] = pd.to_datetime(df['created_utc'], utc=True, unit='s', errors='coerce')
    df['ymd'] = df['datetime'].dt.strftime('%Y-%m-%d')
    df.dropna(subset=['datetime'], inplace=True)
    df.drop(columns=['created_utc'], inplace=True)
    
    df['tidx'] = pd.DatetimeIndex(df['ymd'])
    df.set_index('tidx', inplace=True)
    return df


def sample_discussions(subs: pd.DataFrame, coms: pd.DataFrame, n: int):
    subs = subs.sample(n=n, random_state=42)
    sampled_sub_ids = subs['id'].to_list()
    coms = coms[coms['parent_id'].isin(sampled_sub_ids)]
    return subs, coms
    