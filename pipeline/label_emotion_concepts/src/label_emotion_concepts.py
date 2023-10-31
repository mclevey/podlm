from transformers import pipeline
from podlm.utilities import load_configs, configure_logging, load_parquet, save_parquet
from podlm.text import transformer_emotion_concepts
import logging

# print('🔥🔥🔥 conda activate entities 🔥🔥🔥')

model = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

configure_logging(__file__)
config, _ = load_configs()
subreddits = config['subreddits']

for r in subreddits:
    df = load_parquet(f'{r}_sentences')
    df = transformer_emotion_concepts(df, model, 'sentence', 'id_sentence')    
    save_parquet(df, f'{r}_emotion_concepts')
    logging.debug("Finished labelling emotion concepts.") 