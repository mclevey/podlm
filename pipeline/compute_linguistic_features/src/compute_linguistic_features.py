from podlm.utilities import load_configs, configure_logging, load_parquet, save_parquet
from podlm.text import count_parts_of_speech
import logging
import spacy
nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])

configure_logging(__file__)
config, _ = load_configs()
subreddits = config['subreddits']


for r in subreddits:
    df = load_parquet(f'{r}_sentences')
    df = df.sample(10)
    
    df = count_parts_of_speech(df, nlp, 'sentence') 
    print(df.head())
    logging.debug("Computed linguistic features with spacy.") 
    save_parquet(df, f'{r}_linguistic_features')
    
    