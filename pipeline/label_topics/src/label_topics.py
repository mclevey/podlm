import os
from podlm.utilities import load_configs, configure_logging, load_parquet, save_parquet
from podlm.text import transformer_topics
import logging

# print('🔥🔥🔥 conda activate entities 🔥🔥🔥')

configure_logging(__file__)
config, _ = load_configs()
subreddits = config['subreddits']

sentence_transformer_model = config['sentence_transformer_model']
umap_n_neighbors = config['umap_n_neighbors']
umap_n_components = config['umap_n_components']
hdbscan_min_cluster_size = config['hdbscan_min_cluster_size']
mmr_model_diversity = config['mmr_model_diversity']


for r in subreddits:
    df = load_parquet(f'{r}_sentences')
    # topics, topic_info, topic_model = transformer_topics(df, model="all-MiniLM-L6-v2", textcol='sentence')
    topics, topic_info, topic_model = transformer_topics(df,
                                                         model=sentence_transformer_model, 
                                                         umap_n_neighbors=umap_n_neighbors,
                                                         umap_n_components=umap_n_components,
                                                         hdbscan_min_cluster_size=hdbscan_min_cluster_size,
                                                         mmr_model_diversity=mmr_model_diversity,
                                                         textcol='sentence')
    save_parquet(topics, f'{r}_topics')
    save_parquet(topic_info, f'{r}_topic_info')
    os.makedirs(f'../output/{r}_model_safetensors', exist_ok=True)
    topic_model.save(f'../output/{r}_model_safetensors', serialization="safetensors", save_ctfidf=True)
    logging.debug("Finished labelling topics.")