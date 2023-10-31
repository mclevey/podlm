from podlm.utilities import load_configs, configure_logging, load_parquet, save_parquet
from podlm.text import transformer_sentiment
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging

# print('🔥🔥🔥 conda activate entities 🔥🔥🔥')

model = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
tokenizer = AutoTokenizer.from_pretrained(model, max_len=512)
model = AutoModelForSequenceClassification.from_pretrained(model)

configure_logging(__file__)
config, _ = load_configs()
subreddits = config['subreddits']

for r in subreddits:
    df = load_parquet(f'{r}_sentences')
    df = transformer_sentiment(df, model, tokenizer)
    save_parquet(df, f'{r}_sentiment')
    logging.debug("Finished sentiment analysis.") 