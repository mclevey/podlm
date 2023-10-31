from podlm.utilities import load_configs, configure_logging, load_parquet, save_parquet
from podlm.text import transformer_entities
from span_marker import SpanMarkerModel
import logging

# print('🔥🔥🔥 conda activate entites 🔥🔥🔥')

model = SpanMarkerModel.from_pretrained("lxyuan/span-marker-bert-base-multilingual-uncased-multinerd")
model.cuda()

configure_logging(__file__)
config, _ = load_configs()
subreddits = config['subreddits']
entity_score_threshold = config['entity_score_threshold']


for r in subreddits:
    df = load_parquet(f'{r}_sentences')
    entities, entities_info = transformer_entities(df, model, entity_score_threshold, 'sentence', 'id_sentence')
    save_parquet(entities, f'{r}_entities')
    save_parquet(entities_info, f'{r}_entities_info')
    logging.debug("Constructed entity dataframes (long and counts)") 