from podlm.utilities import load_configs, configure_logging, load_parquet, save_parquet, split_ids
from functools import reduce
import pandas as pd
import logging

configure_logging(__file__)
config, _ = load_configs()
subreddits = config['subreddits']
    
    
for r in subreddits:
    to_merge = []
    tasks = ['emotion_concepts', 'entities', 'sentiment', 'topics', 'linguistic_features'
             ]
    for task in tasks:
        df = load_parquet(f'{r}_{task}')
        df = split_ids(df)
        to_merge.append(df)
        
    merged = reduce(lambda left, right: pd.merge(left, right, on='id_sentence'), to_merge)
    post_level_data = load_parquet(f'{r}_post_level_subcom_merged')
    authors_and_post_ids = post_level_data[['author', 'id']]
    authors_and_post_ids.rename(columns={'id': 'post_id'}, inplace=True)
    merged = pd.merge(merged, authors_and_post_ids, on='post_id')
    save_parquet(merged, f'{r}_sentence_level_features_merged')    
    logging.debug("Completed sentence-level merge.") 