from termcolor import colored
from scipy.special import softmax
from datetime import datetime
import pandas as pd
import spacy
from bertopic import BERTopic
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


def count_categorical(df: pd.DataFrame, id_col: str, cat_col: str, score_threshold: float) -> pd.DataFrame:
    df = df[df['score'] > score_threshold]
    grouped = df.groupby([id_col, cat_col])
    counts = grouped.size().reset_index(name='count')
    pivoted = counts.pivot(index=id_col, columns=cat_col, values='count')
    pivoted = pivoted.fillna(0)
    pivoted = pivoted.reset_index()
    return pivoted


def transformer_entities(df: pd.DataFrame, model, entity_score_threshold: float, textcol: str = 'sentence', idcol: str = 'id_sentence'):
    entity_dfs, errors = [], []
    for i,t in enumerate(df[textcol]):
        try: 
            entities = model.predict(t)
            if len(entities) > 0:
                entdf = pd.DataFrame(entities)
                entdf['id_sentence'] = df.iloc[i][idcol]
                entity_dfs.append(entdf)
        except:
            errors.append([i, t])
    if len(errors) > 0:
        for error in errors:
            print(colored(error, 'red'))
    results_long = pd.concat(entity_dfs)
    # count types above score threshold
    count_types = count_categorical(results_long, 'id_sentence', 'label', score_threshold=entity_score_threshold)
    count_types.columns = [f'entity_count_{c}'.lower() for c in count_types.columns]
    return results_long, count_types


def transformer_topics(df: pd.DataFrame, 
                       model: str = "all-MiniLM-L6-v2", 
                       umap_n_neighbors: int = 15,
                       umap_n_components: int = 5,
                       hdbscan_min_cluster_size: int = 15,
                       mmr_model_diversity: float = 0.3,
                       textcol: str = 'sentence'):
    df = df.reset_index(drop=True)
    sent_embeddings = SentenceTransformer(model)
    embeddings = sent_embeddings.encode(df[textcol], show_progress_bar=True)
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=30)
    hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
    keybert_model = KeyBERTInspired()
    # pos_model = PartOfSpeech("en_core_web_sm")
    mmr_model = MaximalMarginalRelevance(diversity=0.3)
    representation_model = {"KeyBERT": keybert_model, "MMR": mmr_model} # , "POS": pos_model
    topic_model = BERTopic(embedding_model=sent_embeddings, 
                           umap_model=umap_model, 
                           hdbscan_model=hdbscan_model, 
                           vectorizer_model=vectorizer_model, 
                           representation_model=representation_model,
                           top_n_words=10, verbose=True)
    topics, probabilities = topic_model.fit_transform(df[textcol], embeddings)
    df['topic'] = topics
    df['topic_probability'] = probabilities
    topic_info = topic_model.get_topic_info()
    return df, topic_info, topic_model


def transformer_sentiment(df: pd.DataFrame, model, tokenizer, textcol: str = 'sentence', idcol: str = 'id_sentence'):
    scores, errors, sentids = [], [], []
    for i, t in enumerate(df[textcol]):
        try:
            encoded = tokenizer.encode(t, return_tensors='pt')
            sentsent = model(encoded)
            sentsent = sentsent[0][0].detach().numpy()
            sentsent = softmax(sentsent)
            scores.append(sentsent)
            sentids.append(df.iloc[i][idcol])
        except:
            errors.append([i, t])
        if len(errors) > 0:
            for error in errors:
                print(colored(error, 'red'))
    df = pd.DataFrame(scores)
    df.columns = ['sentiment_negative', 'sentiment_neutral', 'sentiment_positive']
    df['sentence_id'] = sentids
    return df
        

def transformer_emotion_concepts(df: pd.DataFrame, model, tokenizer, textcol: str = 'sentence', idcol: str = 'id_sentence'): 
    errors, sentids = [], []
    roberta_base_go_emotions = ['admiration', 'amusement', 'anger', 'annoyance', 
                            'approval', 'caring', 'confusion', 'curiosity', 'desire', 
                            'disappointment', 'disapproval', 'disgust', 'embarrassment', 
                            'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 
                            'nervousness', 'neutral', 'optimism', 'pride', 'realization', 
                            'relief', 'remorse', 'sadness', 'surprise']
    
    ec_wide = []
    for i, t in enumerate(df[textcol]):
        try:
            wide = []
            if len(t) > 512:
                t = truncate_text_to_transformer_limit(text = t)
            emo = model(t)[0] 
            for ec in roberta_base_go_emotions:
                for e in emo:
                    if e['label'] == ec:
                        wide.append(e['score'])
            ec_wide.append(wide)
            sentids.append(df.iloc[i][idcol])
        except:
            errors.append([i, t])
        if len(errors) > 0:
            for error in errors:
                print(colored(error, 'red'))
    df = pd.DataFrame(ec_wide)
    df.columns = roberta_base_go_emotions
    df['sentence_id'] = sentids
    return df


def count_parts_of_speech(df: pd.DataFrame, model, textcol: str = 'sentence') -> pd.DataFrame:
    coarse_counts, ids = [], []
    pronouns_1, pronouns_2, pronouns_3 = [], [], []
    docs = model.pipe(df[textcol])
    for i_doc, doc in enumerate(docs):
        id_sentence = df.iloc[i_doc]['id_sentence']
        ids.append(id_sentence)
        # COARSE POS
        pos_count = doc.count_by(spacy.attrs.POS)
        pos_count = {model.vocab.strings[k]: v for k, v in pos_count.items()}
        coarse_counts.append(pos_count)
        # FINE POS
        first, second, third = categorize_pronouns(doc, model)
        pronouns_1.append(first)
        pronouns_2.append(second)
        pronouns_3.append(third)

    coarse_counts = pd.DataFrame(coarse_counts).fillna(0)
    coarse_counts['id_sentence'] = ids
    coarse_counts['pronoun_first_person'] = pronouns_1
    coarse_counts['pronoun_second_person'] = pronouns_2
    coarse_counts['pronoun_third_person'] = pronouns_3
    coarse_counts['count_pronoun_first_person'] = [len(x) for x in pronouns_1]
    coarse_counts['count_pronoun_second_person'] = [len(x) for x in pronouns_2]
    coarse_counts['count_pronoun_third_person'] = [len(x) for x in pronouns_3]
    return coarse_counts


def matchy_matchy(doc, pattern, model):
    from spacy.matcher import Matcher    
    matcher = Matcher(model.vocab)
    matcher.add('pattern', [pattern])
    matches = matcher(doc, as_spans=True)
    matches = [match.text for match in matches]
    return matches


def categorize_pronouns(doc, model):
    first = matchy_matchy(doc, [{'POS': 'PRON', 'OP': '+', 'MORPH': {'IS_SUPERSET': [f'Person=1']}}], model)
    second = matchy_matchy(doc, [{'POS': 'PRON', 'OP': '+', 'MORPH': {'IS_SUPERSET': [f'Person=2']}}], model)
    third = matchy_matchy(doc, [{'POS': 'PRON', 'OP': '+', 'MORPH': {'IS_SUPERSET': [f'Person=3']}}], model)
    # pronoun_cats = [first, second, third]
    # pronouns = pd.DataFrame(pronoun_cats).T
    # pronouns.columns = ['pronoun_first_person', 'pronoun_second_person', 'pronoun_third_person']
    # pronoun_first_person_count = pronouns['pronoun_first_person'].str.len
    # pronoun_second_person_count = pronouns['pronoun_second_person'].str.len
    # pronoun_third_person_count = pronouns['pronoun_third_person'].str.len
    return first, second, third
        
    
    
    
    # pronoun_first_person = [{'POS': 'PRON', 'OP': '+', 'MORPH': {'IS_SUPERSET': ['Person=1']}}]
    # pronoun_second_person = [{'POS': 'PRON', 'OP': '+', 'MORPH': {'IS_SUPERSET': ['Person=2']}}]
    # pronoun_third_person = [{'POS': 'PRON', 'OP': '+', 'MORPH': {'IS_SUPERSET': ['Person=3']}}]
  
    # first.append(matchy_matchy(doc, pronoun_first_person, model))
    # second.append(matchy_matchy(doc, pronoun_second_person, model))
    # third.append(matchy_matchy(doc, pronoun_third_person, model))




# similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# def llm_cosine_similarity(sentences, ids, positions, llm=similarity_model):
#     results = []
#     paraphrases = util.paraphrase_mining(llm, sentences)
#     for paraphrase in paraphrases:
#         score, i, j = paraphrase
#         results.append([sentences[i], sentences[j], score, ids[i], ids[j], positions[i], positions[j]])
#     df = pd.DataFrame(results)
#     df.columns = ['Sentence[i]', 'Sentence[j]', 'Cosine Similarity', 'Sentence[i] ID', 'Sentence[j] ID', 'Sentence[i] Position', 'Sentence[j] Position']
#     return df
    