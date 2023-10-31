from podlm.utilities import load_configs, configure_logging, load_parquet, save_parquet
from podlm.reddit import sample_discussions
import logging

configure_logging(__file__)
config, _ = load_configs()
subreddits = config['subreddits']
n_conversations = config['sample_n_conversations']

for r in subreddits:
    subs = load_parquet(f'{r}_submissions')
    coms = load_parquet(f'{r}_comments')
    subs, coms = sample_discussions(subs, coms, n=n_conversations)
    logging.debug(f"Sampled {len(subs)} conversations.") 
    save_parquet(subs, f'{r}_submissions')
    save_parquet(coms, f'{r}_comments')