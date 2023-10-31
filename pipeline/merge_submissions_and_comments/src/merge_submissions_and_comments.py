from podlm.utilities import load_configs, configure_logging, load_parquet, save_parquet
from podlm.reddit import merge_submissions_and_comments
import pandas as pd
import logging

configure_logging(__file__)
config, _ = load_configs()
subreddits = config['subreddits']

for r in subreddits:
    subs = load_parquet(f'{r}_submissions') 
    coms = load_parquet(f'{r}_comments')
    df = merge_submissions_and_comments(subs, coms)
    logging.debug("Sucessfuly merged submissions and comments.") 
    save_parquet(df, f'{r}_merged')
     
    
        
    
