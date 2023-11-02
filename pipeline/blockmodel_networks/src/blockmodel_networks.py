from podlm.utilities import load_configs, configure_logging, load_parquet, save_parquet
from podlm.networks import return_graph_type, create_blockmodel, get_graph_data, load_graph
import graph_tool.all as gt
import logging

configure_logging(__file__)
config, _ = load_configs()
subreddits = config['subreddits']
entity_score_threshold = config['entity_score_threshold']


for r in subreddits:
    g = gt.load_graph(f'../input/{r}_author_network.gt')
    # g = load_graph(f'{r}_author_network')
    auth_auth = return_graph_type(g, gtype = 0)
    blockmodel = create_blockmodel(auth_auth)
    gd = get_graph_data(blockmodel)
    save_parquet(gd, f'{r}_blockmodel')
    