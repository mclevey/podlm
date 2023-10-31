from podlm.utilities import load_configs, configure_logging, load_parquet, save_parquet
from podlm.text import split_sentences
import pandas as pd
import logging
import spacy
nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])
nlp.add_pipe('sentencizer') 

configure_logging(__file__)
config, _ = load_configs()
subreddits = config['subreddits']

for r in subreddits:
    df = load_parquet(f'{r}_merged')
    df = split_sentences(df, nlp, 'text', 'id')
    logging.debug("Extracted sentences from posts and assigned sentence-level ids.") 
    save_parquet(df, f'{r}_sentences')