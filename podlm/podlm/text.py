from termcolor import colored
from scipy.special import softmax
from datetime import datetime
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech

def split_sentences(df: pd.DataFrame, model, textcol: str = 'text', idcol: str = 'id') -> pd.DataFrame:
    sentences = []
    texts = df[textcol].replace(r'\n',' ', regex=True)
    docs = model.pipe(texts)
    for i_doc, doc in enumerate(docs):
        for i_sent, sent in enumerate(doc.sents):
            if len(sent) > 2:
                post_id = df.iloc[i_doc][idcol] 
                id_sentence = str(post_id) + '_' + str(i_sent)
                sentences.append({'id_sentence': id_sentence, 'sentence': sent.text})
    sentences = pd.DataFrame(sentences)    
    return sentences


def transformer_sentiment(df, idcol, textcol, poscol, authorcol, datecol, tokenizer, model):
    scores, ids, positions, author_l, dates_l, sent_l, errors = [], [], [], [], [], [], []
    id_list = df[idcol].tolist() 
    sent_list = df[textcol].tolist() 
    posids = df[poscol].tolist()
    authors = df[authorcol].tolist()
    dates = df[datecol].tolist()
    
    for i, sent, pos, a, d in zip(id_list, sent_list, posids, authors, dates):
        try:
            encoded = tokenizer.encode(sent, return_tensors='pt')
            sentsent = model(encoded)
            sentsent = sentsent[0][0].detach().numpy()
            sentsent = softmax(sentsent) # convert to probabilities
            scores.append(sentsent)
            ids.append(i)
            positions.append(pos)
            author_l.append(a)
            dates_l.append(d)
            sent_l.append(sent)
        except:
            errors.append(f'Unable to process {i}\n{sent}\n\n')
    df = pd.DataFrame(scores)
    df.columns = ['p(positive) sentiment', 'p(neutral) sentiment', 'p(negative) sentiment']
    df['id'] = ids
    df['sentence_position_in_post'] = positions
    df['author'] = author_l
    df['date'] = dates_l
    df['sentence'] = sent_l
        
    if len(errors) > 0:
        with open('../output/errors.txt', 'a', encoding='utf-8') as f:
            for e in errors:
                now = datetime.now()
                now_str = now.strftime("%d/%m/%Y %H:%M:%S")
                f.write(f'Errors from run: {now_str}\n')
                f.write(e + '\n')
    return df


def transformer_entities(df, textcol, idcol, poscol, authorcol, datecol, model):
    entity_dfs, errors = [], []
    
    text = df[textcol].to_list()
    ids = df[idcol].to_list()
    posids = df[poscol].to_list()
    authors = df[authorcol].to_list()
    dates = df[datecol].to_list()

    for t,i,p,a,d in zip(text, ids, posids, authors, dates):
        try: 
            entities = model.predict(t.strip())
            if len(entities) > 0:
                entdf = pd.DataFrame(entities)
                entdf['id'] = i
                entdf['sentence_position_in_post'] = p
                entdf['sentence'] = t
                entdf['date'] = d
                entdf['author'] = a
                entity_dfs.append(entdf)
        except:
            errors.append([i, p, t])
    results = pd.concat(entity_dfs)
    results.rename(columns={'span': 'entity_span',
                   'label': 'entity_label',
                   'score': 'entity_score',
                   'char_start_index': 'entity_start_index',
                   'char_end_index': 'entity_char_end_index'
                   }, inplace=True)
    return results, errors


def transformer_topics(df: pd.DataFrame, textcol: str, 
                       sentence_transformer_model: str, 
                       hdbscan_min_cluster_size=15, 
                       show_sent_embeddings_progress_bar=True):
    """
    This function executes a BERTopic pipeline one step at a time so that I can easily
    configure the parameters of each modular component. 
    
    1. Compute sentence embeddings in advance and then pass them to the BERTopic model
    2. Use UMAP to reduce the dimensionality of the sentence embeddings
    3. Use HDBSCAN to cluster the sentence embeddings
    4. Re-parameterize CountVectorizer and compute ctf-idf scores for each sentence
    5. Compute additional topic representations using KeyBERTInspired, MaximalMarginalRelevance, and PartOfSpeech models
    6. Fit the topic model!
    """
    from bertopic import BERTopic

    text = df[textcol].tolist()    
    sent_embeddings = SentenceTransformer(sentence_transformer_model)
    embeddings = sent_embeddings.encode(text, show_progress_bar=show_sent_embeddings_progress_bar)
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=30)
    hdbscan_model = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
    keybert_model = KeyBERTInspired()
    pos_model = PartOfSpeech("en_core_web_sm")
    mmr_model = MaximalMarginalRelevance(diversity=0.3)
    representation_model = {"KeyBERT": keybert_model, "MMR": mmr_model, "POS": pos_model}
    
    topic_model = BERTopic(embedding_model=sent_embeddings, umap_model=umap_model, hdbscan_model=hdbscan_model, 
                           vectorizer_model=vectorizer_model, representation_model=representation_model,
                           top_n_words=10, verbose=True)

    topics, probabilities = topic_model.fit_transform(df[textcol].tolist(), embeddings)

    topic_info = topic_model.get_topic_info()
    df['topic'] = topics
    df['topic_probability'] = probabilities
    return topics, probabilities, topic_info, df, topic_model


def transformer_emotion_concepts(df, idcol, poscol, authorcol, datecol, textcol, model):
    errors = []
    roberta_base_go_emotions = ['admiration', 'amusement', 'anger', 'annoyance', 
                            'approval', 'caring', 'confusion', 'curiosity', 'desire', 
                            'disappointment', 'disapproval', 'disgust', 'embarrassment', 
                            'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 
                            'nervousness', 'neutral', 'optimism', 'pride', 'realization', 
                            'relief', 'remorse', 'sadness', 'surprise']
    
    meta, ec_wide = [], []
    for i,p,a,d,t in zip(df[idcol], df[poscol], df[authorcol], df[datecol], df[textcol]):
        t = t.strip() # this should be done early in the project pipeline so it applies everywhere
        try:
            wide = []
            meta.append([i, p, a, d, t])
            # truncate to 512 tokens, doesn't really matter since we are working at sent-level
            if len(t) > 512:
                t = truncate_text_to_transformer_limit(text = t)
            emo = model(t)[0] 
            for ec in roberta_base_go_emotions:
                for e in emo:
                    if e['label'] == ec:
                        wide.append(e['score'])
            ec_wide.append(wide)
        except:
            errors.append(f'Unable to process {i}\n{t}\n\n')
        
    results = [m + e for m, e in zip(meta, ec_wide)]
    results = pd.DataFrame(results)    
    results.columns = ['id', 'sentence_position_in_post', 'author', 'date', 'sentence'] + roberta_base_go_emotions   
    return results, errors


def truncate_text_to_transformer_limit(text: str, transformer_token_limit: int = 512) -> str:
    original_text = text
    truncated_text = text[:512]
    cutoff_tokens = text[512:]
    overlimit = len(original_text) - transformer_token_limit
    boundary = '------------------------------------------------------------------'
    message_p1 = f'The following text is {overlimit} tokens over the 512 limit:'
    message_p2 = 'Truncating to 512 tokens to avoid transformer error or indexing problems. The truncated version is as follows:'
    message_p3 = 'The following tokens did not get passed to the model:'
    print(colored(boundary, 'red'))
    print(colored(message_p1, 'red'))
    print(original_text)
    print(colored(message_p2, 'red'))
    print(truncated_text)
    print(colored(message_p3, 'red'))
    print(cutoff_tokens)
    print(colored(boundary, 'red'))
    return text


# def matchy_matchy(doc, pattern, model):
#     from spacy.matcher import Matcher    
#     matcher = Matcher(model.vocab)
#     matcher.add('pattern', [pattern])
#     matches = matcher(doc, as_spans=True)
#     matches = [match.text for match in matches]
#     return matches

def count_parts_of_speech(df: pd.DataFrame, model, textcol: str = 'sentence') -> pd.DataFrame:
    counts, ids = [], []
    docs = model.pipe(df[textcol])
    for i_doc, doc in enumerate(docs):
        id_sentence = df.iloc[i_doc]['id_sentence']
        ids.append(id_sentence)
        # COARSE POS
        pos_count = doc.count_by(spacy.attrs.POS)
        pos_count = {nlp.vocab.strings[k]: v for k, v in pos_count.items()}
        counts.append(pos_count)
        
    counts = pd.DataFrame(counts).fillna(0)
    counts['id_sentence'] = ids
    return counts








"""
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm




similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

def llm_cosine_similarity(sentences, ids, positions, llm=similarity_model):
    results = []
    paraphrases = util.paraphrase_mining(llm, sentences)
    for paraphrase in paraphrases:
        score, i, j = paraphrase
        results.append([sentences[i], sentences[j], score, ids[i], ids[j], positions[i], positions[j]])
    df = pd.DataFrame(results)
    df.columns = ['Sentence[i]', 'Sentence[j]', 'Cosine Similarity', 'Sentence[i] ID', 'Sentence[j] ID', 'Sentence[i] Position', 'Sentence[j] Position']
    return df
    
"""