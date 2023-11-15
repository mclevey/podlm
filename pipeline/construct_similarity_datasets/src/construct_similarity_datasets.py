from podlm.utilities import load_configs, configure_logging, load_parquet, save_parquet
from podlm.networks import load_gt, get_graph_dfs

configure_logging(__file__)
config, _ = load_configs()
subreddits = config["subreddits"]


for r in subreddits:
    g = load_gt(f"{r}_author_network")
    vdf, edf = get_graph_dfs(g, vtype=0, etype=0)
    save_parquet(vdf, f"{r}_vertices")
    save_parquet(edf, f"{r}_edges")
