from elasticsearch import Elasticsearch
from podlm.utilities import load_configs, configure_logging, save_parquet
from podlm.reddit import es_query_reddit
import logging

configure_logging(__file__)
config, private = load_configs()
subreddits = config['subreddits']
es = Elasticsearch(private['es_host'], verify_certs = private['es_verify_certs'])

for r in subreddits:
    subs, coms = es_query_reddit(r, es)
    logging.debug(f"ES query returned {len(subs)} submissions and {len(coms)} comments.") 
    save_parquet(subs, f'{r}_submissions')
    save_parquet(coms, f'{r}_comments')