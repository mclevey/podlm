from podlm.utilities import load_configs, configure_logging, load_parquet, save_parquet
from functools import reduce
import pandas as pd
import logging

configure_logging(__file__)
config, _ = load_configs()
subreddits = config['subreddits']
    
for r in subreddits:
    to_merge = []
    tasks = [
        'emotion_concepts',
        'entities',
        'sentiment',
        'topics',
        'linguistic_features'
        ]
    for task in tasks:
        df = load_parquet(f'{r}_{task}')
        print(df.columns)
        to_merge.append(df)    
    merged = reduce(lambda left, right: pd.merge(left, right, on='id_sentence'), to_merge)
    save_parquet(merged, f'{r}_sentence_level_features_merged')    
    logging.debug("Completed sentence-level merge.") 