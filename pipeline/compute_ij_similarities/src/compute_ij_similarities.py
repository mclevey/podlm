from podlm.utilities import load_configs, configure_logging, load_parquet, save_parquet

from sentence_transformers import SentenceTransformer
from podlm.text import compute_ij_cosine_similarity_before_datetime

configure_logging(__file__)
config, _ = load_configs()
subreddits = config["subreddits"]

similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

for r in subreddits:
    vdf = load_parquet(f"{r}_vertices")
    edf = load_parquet(f"{r}_edges")

    tdf = load_parquet(f"{r}_sentence_level_features_merged")
    tdf = tdf[["author", "sentence", "datetime"]]

    edf_csim = compute_ij_cosine_similarity_before_datetime(
        edf, tdf, model=similarity_model
    )
    save_parquet(edf_csim, f"{r}_ij_similarity")
