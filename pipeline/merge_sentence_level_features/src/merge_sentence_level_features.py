from podlm.utilities import load_configs, configure_logging, load_parquet, save_parquet, split_ids
from functools import reduce
import pandas as pd
import logging

configure_logging(__file__)
config, _ = load_configs()
subreddits = config['subreddits']
    
    
for r in subreddits:
    to_merge = []
    tasks = ['emotion_concepts', 'entities', 'sentiment', 'topics', 'linguistic_features']
    for task in tasks:
        df = load_parquet(f'{r}_{task}')
        df = split_ids(df)
        to_merge.append(df)        
    merged = reduce(lambda left, right: pd.merge(left, right, on='id_sentence'), to_merge)
     
    post_level_data = load_parquet(f'{r}_post_level_subcom_merged')
     
    authors_and_post_ids = post_level_data[[
        'author', 'id', 'parent_id', 'datetime', 'subreddit', 
        'is_submitter', 'score', 'num_comments', 'url', 'domain'
        ]].copy()
    
    authors_and_post_ids.loc[:, 'post_id'] = authors_and_post_ids['id']
    authors_and_post_ids.drop('id', axis=1, inplace=True)
    authors_and_post_ids.reset_index(drop=True, inplace=True)
    with_authors = pd.merge(authors_and_post_ids, merged, on='post_id')    
    dropcols = [c for c in with_authors.columns if c.endswith('_x') or c.endswith('_y')]
    for dc in dropcols:
        try:
            with_authors.drop(dc, axis=1, inplace=True)
        except:
            pass
    save_parquet(with_authors, f'{r}_sentence_level_features_merged')
    logging.debug("Completed sentence-level merge.") 