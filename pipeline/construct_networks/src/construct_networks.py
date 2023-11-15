from podlm.utilities import load_configs, configure_logging, load_parquet
from podlm.networks import construct_network, save_gt
import logging

configure_logging(__file__)
config, _ = load_configs()
subreddits = config["subreddits"]


for r in subreddits:
    df = load_parquet(f"{r}_sentence_level_features_merged")
    g = construct_network(df, "author", "post_id", "parent_id")
    save_gt(g, f"{r}_author_network")
    logging.debug("Constructed entity dataframes (long and counts)")
