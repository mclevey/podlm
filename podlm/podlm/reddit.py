import pandas as pd
import urllib3
urllib3.disable_warnings()
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from podlm.utilities import merge_string_columns, to_POSIX, from_POSIX

""" OLD...
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
"""

def es_query_reddit(search: str, es: Elasticsearch):
    results = {}
    query = {'query': {'match': {'subreddit': search}}}
    reddit_indexes = ['reddit-index-s', 'reddit-index-c']
    for ri in reddit_indexes:
        search_results = scan(es, query = query, index=ri)
        df = es_results_to_df(search_results)

        # FIX: Rename _id to id before dropping other columns
        if '_id' in df.columns:
            df.rename(columns={'_id': 'id'}, inplace=True)

        # FIX: Only drop columns that actually exist (excluding _id which is now renamed)
        dropcols = ['_source', '_score', '_index', 'gildings', 'sort']
        existing_dropcols = [col for col in dropcols if col in df.columns]
        if existing_dropcols:
            df.drop(columns=existing_dropcols, inplace=True)

        df = set_dtypes(df)
        df = process_datetimes(df)

        if ri == 'reddit-index-s':
            df['id'] = 't3_' + df['id']
            results['subs'] = df
        elif ri == 'reddit-index-c':
            df['id'] = 't1_' + df['id']
            results['coms'] = df
    return results['subs'], results['coms']

# NEW (roughly 3x faster)
def es_results_to_df(search_results):
    df = pd.json_normalize(data = search_results)
    df.columns = [col.split(".")[-1] for col in df.columns]
    df['created_utc'] = df['created_utc'].astype(int)
    
    # FIX: Convert ID columns to strings to avoid mixed types
    id_columns = [c for c in df.columns if '_id' in c]
    for col in id_columns:
        df[col] = df[col].astype(str)

    return df


def set_dtypes(df: pd.DataFrame):
    df['subreddit_type'] = df['subreddit_type'].astype('category')
    if 'gilded' in df.columns:
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
    
    
def merge_submissions_and_comments(submissions: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    submissions = merge_string_columns(submissions, 'selftext', 'title', 'text', drop=True)
    submissions['parent_id'] = submissions['id']
    comments.rename(columns={'body': 'text'}, inplace=True)
    df = pd.concat([submissions, comments])
    return df


### NEW QUERY FUNCTIONS


def es_run_query_reddit(query: dict, es: Elasticsearch, search_index: str = 'reddit-index-s'):
    all_resp = scan(es, query = query, index= search_index)    
    return all_resp
    

def es_construct_query(exact_terms = None,
                           search_terms_all = None,
                           search_terms_any = None,
                           search_phrases_all = None,
                           search_phrases_any = None,
                           date_start = None,
                           date_end = None):

    '''
    First five options expect a list of 2 item lists. The 2 item lists are field-string pairs. 
    
    Examples:
    
    exact_terms_to_search = [['subreddit', 'politics']]
    search_terms_all = [['title', 'MAGA'],['domain','breitbart.com']]

    
    
    date_start and date_end expect a list of integers [YYYY, MM, DD]
    
    Example to search first day of January, 2017:
    
    date_start = [2017,01,01]
    date_end = [2017,01,02]
    
    '''

    
    search_query = {'query': {'bool': {}}}
    date_range = {'range': {'created_utc': {'format':'epoch_second'}}}
    
    filter_list = []    
    must_match_list = []    
    should_match_list = []
    
    
    if date_start:
        start = to_POSIX(date_start)
        date_range['range']['created_utc'].update({'gte':start})
    
    if date_end:
        end = to_POSIX(date_end)
        date_range['range']['created_utc'].update({'lt':end})
        
    if date_start or date_end:
        filter_list.append(date_range)        
    
    if exact_terms:
        for term in exact_terms:
            term_dict = {'term': {term[0]: {'value': term[1], 'case_insensitive': True}}}
            filter_list.append(term_dict)            
            
    if date_start or date_end or exact_terms:
        search_query['query']['bool'].update({'filter':filter_list})
        
        
    if search_terms_all:
        for term in search_terms_all:
            term_dict = {'match': {term[0]: {'query': term[1], 'fuzziness': 'AUTO'}}}
            must_match_list.append(term_dict)
            
    if search_phrases_all:
        for term in search_phrases_all:
            term_dict = {'match_phrase': {term[0]: term[1]}}
            must_match_list.append(term_dict)
            
    if search_terms_all or search_phrases_all:
        search_query['query']['bool'].update({'must': must_match_list})
        

    if search_terms_any:
        for term in search_terms_any:
            term_dict = {'match': {term[0]: {'query': term[1], 'fuzziness': 'AUTO'}}}
            should_match_list.append(term_dict)   
            
    if search_phrases_any:
        for term in search_phrases_any:
            term_dict = {'match_phrase': {term[0]: term[1]}}
            should_match_list.append(term_dict) 
            
    if search_terms_any or search_phrases_any:
        search_query['query']['bool'].update({'should': should_match_list})
        search_query['query']['bool'].update({'minimum_should_match': 1})
        
    
    return search_query
