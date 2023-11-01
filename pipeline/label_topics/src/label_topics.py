import os
from podlm.utilities import load_configs, configure_logging, load_parquet, save_parquet
from podlm.text import transformer_topics
import logging

# print('🔥🔥🔥 conda activate entities 🔥🔥🔥')

configure_logging(__file__)
config, _ = load_configs()
subreddits = config['subreddits']

for r in subreddits:
    df = load_parquet(f'{r}_sentences')
    topics, topic_info, topic_model = transformer_topics(df, model="all-MiniLM-L6-v2", textcol='sentence', idcol='id_sentence')
    save_parquet(topics, f'{r}_topics')
    save_parquet(topic_info, f'{r}_topic_info')
    os.makedirs(f'../output/{r}_model_safetensors', exist_ok=True)
    topic_model.save(f'../output/{r}_model_safetensors', serialization="safetensors", save_ctfidf=True)
    logging.debug("Finished labelling topics.")