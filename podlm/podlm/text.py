from termcolor import colored
from scipy.special import softmax
from datetime import datetime
import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech


def split_sentences(df: pd.DataFrame, textcol: str, idcol: str, datetimecol, authorcol: str, model):
    sentences, positions, ids, dates, authors = [], [], [], [], []
    texts = df[textcol].tolist()
    # for some reason, t.strip() is not stripping all the \n characters, so I'm doing it manually...
    texts = [t.strip().replace('\n', '') for t in texts if isinstance(t, str) is True]    
    docs = model.pipe(texts)
    for didx, doc in enumerate(docs):
        for sidx, sent in enumerate(doc.sents):
            if len(sent) > 2:
                sentences.append(" ".join([token.text for token in sent])) 
                positions.append(sidx)
                ids.append(df[idcol].iloc[didx])
                dates.append(df[datetimecol].iloc[didx])
                authors.append(df[authorcol].iloc[didx])
    df = pd.DataFrame([sentences, positions, ids, dates, authors]).T
    df.columns = ["sentence", "sentence_position_in_post", 'id', 'date', 'author']
    return df


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


def matchy_matchy(doc, pattern, model):
    from spacy.matcher import Matcher    
    matcher = Matcher(model.vocab)
    matcher.add('pattern', [pattern])
    matches = matcher(doc, as_spans=True)
    return [match.text for match in matches]


def find_future_tense_sentences(doc, model):
    auxiliary_verbs_indicating_future_tense = ['will', 'shall', 'going', 'about', 'intend', 'aim', 'plan', 'expect', 'hope', 'would', 'like', 'likely', 'bound', 'scheduled', 'due', 'set']
    future_tense_pattern = [
        {'POS': 'AUX', 'LOWER': {'IN': auxiliary_verbs_indicating_future_tense}},
        {'POS': 'VERB'}
    ] 
    matches = matchy_matchy(doc, future_tense_pattern, model)
    # matcher = Matcher(model.vocab)
    # matcher.add('FUTURE_TENSE', [future_tense_pattern])
    # matches = matcher(doc, as_spans=True)
    # matches = [match.text for match in matches]
    len_matches = [len(m) for m in matches]
    if len_matches != []:
        return True, matches
    else:
        return False, matches
    
def subsentence_processing_pipeline(df, textcol, model):
    docs = model.pipe(df[textcol].tolist())
    
    verbs_present, verbs_past = [], []
    future_tense, future_tense_matches = [], []
    pronouns_first, pronouns_second, pronouns_third = [], [], []
    basic_text_features = []
    
    for doc in docs:
        basic_text_features.append(len(doc))
        # MATCH VERBS
        verbs_present_tense = [{'POS': 'VERB', 'OP': '+', 'MORPH': {'IS_SUPERSET': ['Tense=Pres']}}]
        verbs_past_tense = [{'POS': 'VERB', 'OP': '+', 'MORPH': {'IS_SUPERSET': ['Tense=Past']}}]
                
        verbs_present.append(matchy_matchy(doc, verbs_present_tense, model))
        verbs_past.append(matchy_matchy(doc, verbs_past_tense, model))        
           
        # COULD THIS BE A FUTURE TENSE SENTENCE?
        future, future_matches = find_future_tense_sentences(doc, model)   
        future_tense.append(future)
        future_tense_matches.append(future_matches)
           
        # MATCH PRONOUNS     
        pronoun_first_person = [{'POS': 'PRON', 'OP': '+', 'MORPH': {'IS_SUPERSET': ['Person=1']}}]
        pronoun_second_person = [{'POS': 'PRON', 'OP': '+', 'MORPH': {'IS_SUPERSET': ['Person=2']}}]
        pronoun_third_person = [{'POS': 'PRON', 'OP': '+', 'MORPH': {'IS_SUPERSET': ['Person=3']}}]
        
        pronouns_first.append(matchy_matchy(doc, pronoun_first_person, model))
        pronouns_second.append(matchy_matchy(doc, pronoun_second_person, model))
        pronouns_third.append(matchy_matchy(doc, pronoun_third_person, model))
        
    # CREATE RESULTS DATAFRAME
    results = pd.DataFrame()
    results['id'] = df['id']
    results['author'] = df['author']
    results['sentence'] = df['sentence']
    results['sentence_position_in_post'] = df['sentence_position_in_post']
    results['date'] = df['date']
    
    # ADD VERBS
    results['verbs_present_tense'] = verbs_present
    results['count_verbs_present_tense'] = [len(l) for l in verbs_present]
    results['verbs_past_tense'] = verbs_past
    results['count_verbs_past_tense'] = [len(l) for l in verbs_past]
    results['bool_sent_likely_future_tense'] = future_tense
    results['aux_+_verb_future_tense'] = future_tense_matches
    
    # ADD PRONOUNS
    results['pronouns_first'] = pronouns_first
    results['count_pronouns_first'] = [len(l) for l in pronouns_first]
    results['pronouns_second'] = pronouns_second
    results['count_pronouns_second'] = [len(l) for l in pronouns_second]
    results['pronouns_third'] = pronouns_third
    results['count_pronouns_third'] = [len(l) for l in pronouns_third]
    
    results['number_of_tokens'] = basic_text_features
    return results

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