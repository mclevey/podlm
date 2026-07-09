from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.special import softmax
except ImportError:

    def softmax(values: Sequence[float]) -> np.ndarray:
        """Return a softmax vector when SciPy is unavailable."""

        values_array = np.asarray(values, dtype=float)
        exp_values = np.exp(values_array - np.max(values_array))
        return exp_values / exp_values.sum()


GO_EMOTIONS_LABELS: tuple[str, ...] = (
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "neutral",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
)


def _raise_labeling_errors(
    task_name: str, errors: list[tuple[object, Exception]]
) -> None:
    """Raise one compact error for row-level model inference failures."""

    if not errors:
        return
    examples = "; ".join(
        f"{row_id}: {type(error).__name__}: {error}"
        for row_id, error in errors[:5]
    )
    raise RuntimeError(
        f"{task_name} failed for {len(errors)} row(s). "
        f"First failures: {examples}"
    )


def _tensor_to_array(value: Any) -> np.ndarray:
    """Convert torch-like tensors and array-like values to a NumPy array."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)


def _encode_text(tokenizer: Any, text: str, max_length: int) -> Any:
    """Encode text with tokenizer truncation and a legacy-tokenizer fallback."""

    try:
        return tokenizer.encode(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
    except TypeError:
        return tokenizer.encode(
            truncate_text_to_transformer_limit(text, max_length),
            return_tensors="pt",
        )


def split_sentences(
    df: pd.DataFrame, model: Any, textcol: str = "text", idcol: str = "id"
) -> pd.DataFrame:
    """Split post text into sentence rows without losing source-row alignment.

    Parameters
    ----------
    df
        Input frame with post identifiers and text.
    model
        spaCy-like model exposing ``pipe`` and returning docs with ``sents``.
    textcol
        Column containing post text.
    idcol
        Column containing post identifiers.

    Returns
    -------
    pd.DataFrame
        Sentence-level frame with ``id_sentence`` and ``sentence`` columns.

    Raises
    ------
    KeyError
        If required columns are absent.
    """

    source = df[[idcol, textcol]].reset_index(drop=True)
    texts: list[str] = []
    source_rows: list[int] = []
    for row_index, value in enumerate(source[textcol].tolist()):
        if not isinstance(value, str):
            continue
        text = value.strip().replace("\n", " ")
        if not text:
            continue
        texts.append(text)
        source_rows.append(row_index)

    sentence_rows: list[dict[str, object]] = []
    for source_row, doc in zip(source_rows, model.pipe(texts)):
        post_id = source.iloc[source_row][idcol]
        for sentence_index, sentence in enumerate(doc.sents):
            if len(sentence) > 2:
                sentence_rows.append(
                    {
                        "id_sentence": f"{post_id}_{sentence_index}",
                        "sentence": sentence.text,
                    }
                )
    return pd.DataFrame(sentence_rows, columns=["id_sentence", "sentence"])


def count_categorical(
    df: pd.DataFrame, id_col: str, cat_col: str, score_threshold: float
) -> pd.DataFrame:
    """Count category labels above a score threshold for each identifier.

    Parameters
    ----------
    df
        Long-format classifier output.
    id_col
        Identifier column.
    cat_col
        Category label column.
    score_threshold
        Minimum score required for a label to be counted.

    Returns
    -------
    pd.DataFrame
        Wide count frame with one row per identifier.

    Raises
    ------
    KeyError
        If required columns are absent.
    """

    filtered = df[df["score"] > score_threshold]
    if filtered.empty:
        return pd.DataFrame(columns=[id_col])
    grouped = filtered.groupby([id_col, cat_col])
    counts = grouped.size().reset_index(name="count")
    pivoted = counts.pivot(index=id_col, columns=cat_col, values="count")
    pivoted = pivoted.fillna(0)
    pivoted = pivoted.reset_index()
    return pivoted


def transformer_entities(
    df: pd.DataFrame,
    model: Any,
    entity_score_threshold: float,
    textcol: str = "sentence",
    idcol: str = "id_sentence",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label named entities while preserving sentence identifiers.

    Parameters
    ----------
    df
        Sentence-level frame.
    model
        SpanMarker-like model exposing ``predict``.
    entity_score_threshold
        Minimum score for entity count summaries.
    textcol
        Text column to label.
    idcol
        Sentence identifier column.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Long entity output and wide entity-count output.

    Raises
    ------
    RuntimeError
        If any row fails during model inference.
    """

    entity_dfs: list[pd.DataFrame] = []
    errors: list[tuple[object, Exception]] = []
    for row_id, text in zip(
        df[idcol].tolist(), df[textcol].tolist(), strict=True
    ):
        try:
            entities = model.predict(text)
            if entities:
                entity_df = pd.DataFrame(entities)
                entity_df["id_sentence"] = row_id
                entity_dfs.append(entity_df)
        except Exception as error:
            errors.append((row_id, error))
    _raise_labeling_errors("entity labeling", errors)

    if not entity_dfs:
        return pd.DataFrame(columns=["id_sentence"]), pd.DataFrame(
            columns=["id_sentence"]
        )

    results_long = pd.concat(entity_dfs, ignore_index=True)
    count_types = count_categorical(
        results_long,
        "id_sentence",
        "label",
        score_threshold=entity_score_threshold,
    )
    count_types = count_types.rename(
        columns={
            column: f"entity_count_{column}".lower()
            for column in count_types.columns
            if column != "id_sentence"
        }
    )
    return results_long, count_types


def transformer_topics(
    df: pd.DataFrame,
    model: str = "all-MiniLM-L6-v2",
    umap_n_neighbors: int = 15,
    umap_n_components: int = 5,
    hdbscan_min_cluster_size: int = 15,
    mmr_model_diversity: float = 0.3,
    textcol: str = "sentence",
    include_pos_representation: bool = False,
    show_sent_embeddings_progress_bar: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    """Fit a modular BERTopic model with explicit component parameters.

    Parameters
    ----------
    df
        Sentence-level frame.
    model
        SentenceTransformer model name.
    umap_n_neighbors
        UMAP neighbor count.
    umap_n_components
        UMAP output dimensionality.
    hdbscan_min_cluster_size
        HDBSCAN minimum cluster size.
    mmr_model_diversity
        MMR diversity parameter.
    textcol
        Text column to model.
    include_pos_representation
        Whether to include BERTopic's POS representation model. This restores
        the useful `odlm` option without making it mandatory for every run.
    show_sent_embeddings_progress_bar
        Whether SentenceTransformers should show its progress bar.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, Any]
        Sentence frame with topic assignments, topic info, and fitted model.

    Raises
    ------
    ImportError
        If optional topic-modeling dependencies are not installed.
    """

    from bertopic import BERTopic
    from bertopic.representation import (
        KeyBERTInspired,
        MaximalMarginalRelevance,
        PartOfSpeech,
    )
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    topic_df = df.reset_index(drop=True).copy()
    sent_embeddings = SentenceTransformer(model)
    texts = topic_df[textcol].tolist()
    embeddings = sent_embeddings.encode(
        texts, show_progress_bar=show_sent_embeddings_progress_bar
    )
    umap_model = UMAP(
        n_neighbors=umap_n_neighbors,
        n_components=umap_n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=30,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=hdbscan_min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        stop_words="english", min_df=2, ngram_range=(1, 2)
    )
    representation_model = {
        "KeyBERT": KeyBERTInspired(),
        "MMR": MaximalMarginalRelevance(diversity=mmr_model_diversity),
    }
    if include_pos_representation:
        representation_model["POS"] = PartOfSpeech("en_core_web_sm")

    topic_model = BERTopic(
        embedding_model=sent_embeddings,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        top_n_words=10,
        verbose=True,
    )
    topics, probabilities = topic_model.fit_transform(texts, embeddings)
    topic_df["topic"] = topics
    topic_df["topic_probability"] = probabilities
    topic_info = topic_model.get_topic_info()
    return topic_df, topic_info, topic_model


def transformer_sentiment(
    df: pd.DataFrame,
    model: Any,
    tokenizer: Any,
    textcol: str = "sentence",
    idcol: str = "id_sentence",
    max_length: int = 512,
) -> pd.DataFrame:
    """Score sentiment for each sentence with fail-loud error handling.

    Parameters
    ----------
    df
        Sentence-level frame.
    model
        Transformers sequence classification model.
    tokenizer
        Matching tokenizer.
    textcol
        Text column to classify.
    idcol
        Sentence identifier column.
    max_length
        Tokenizer maximum length.

    Returns
    -------
    pd.DataFrame
        Sentiment probabilities keyed by ``id_sentence``.

    Raises
    ------
    RuntimeError
        If any row fails during model inference.
    """

    rows: list[dict[str, object]] = []
    errors: list[tuple[object, Exception]] = []
    for row_id, text in zip(
        df[idcol].tolist(), df[textcol].tolist(), strict=True
    ):
        try:
            encoded = _encode_text(tokenizer, text, max_length)
            scores = softmax(_tensor_to_array(model(encoded)[0][0]))
            rows.append(
                {
                    "id_sentence": row_id,
                    "sentiment_negative": scores[0],
                    "sentiment_neutral": scores[1],
                    "sentiment_positive": scores[2],
                }
            )
        except Exception as error:
            errors.append((row_id, error))
    _raise_labeling_errors("sentiment labeling", errors)
    return pd.DataFrame(
        rows,
        columns=[
            "id_sentence",
            "sentiment_negative",
            "sentiment_neutral",
            "sentiment_positive",
        ],
    )


def transformer_emotion_concepts(
    df: pd.DataFrame,
    model: Any,
    textcol: str = "sentence",
    idcol: str = "id_sentence",
    max_length: int = 512,
) -> pd.DataFrame:
    """Score GoEmotions labels for each sentence.

    Parameters
    ----------
    df
        Sentence-level frame.
    model
        Hugging Face pipeline created with ``top_k=None``.
    textcol
        Text column to classify.
    idcol
        Sentence identifier column.
    max_length
        Character fallback length used before model inference.

    Returns
    -------
    pd.DataFrame
        Wide GoEmotions score frame keyed by ``id_sentence``.

    Raises
    ------
    RuntimeError
        If any row fails or if model output misses expected labels.
    """

    rows: list[dict[str, object]] = []
    errors: list[tuple[object, Exception]] = []
    for row_id, text in zip(
        df[idcol].tolist(), df[textcol].tolist(), strict=True
    ):
        try:
            if not isinstance(text, str):
                raise TypeError("sentence text must be a string")
            model_text = truncate_text_to_transformer_limit(text, max_length)
            emotion_scores = {
                item["label"]: item["score"]
                for item in model(model_text)[0]
            }
            missing = [
                label for label in GO_EMOTIONS_LABELS if label not in emotion_scores
            ]
            if missing:
                raise ValueError(f"missing emotion labels: {', '.join(missing)}")
            output_row: dict[str, object] = {"id_sentence": row_id}
            output_row.update(
                {label: emotion_scores[label] for label in GO_EMOTIONS_LABELS}
            )
            rows.append(output_row)
        except Exception as error:
            errors.append((row_id, error))
    _raise_labeling_errors("emotion concept labeling", errors)
    return pd.DataFrame(rows, columns=["id_sentence", *GO_EMOTIONS_LABELS])


def truncate_text_to_transformer_limit(
    text: str, transformer_token_limit: int = 512
) -> str:
    """Return a bounded text string without printing sensitive user content.

    Parameters
    ----------
    text
        Input text.
    transformer_token_limit
        Conservative character bound used when tokenizer-level truncation is
        unavailable.

    Returns
    -------
    str
        Original text if it is within the bound, otherwise the truncated text.

    Raises
    ------
    ValueError
        If the limit is less than 1.
    TypeError
        If text is not a string.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if transformer_token_limit < 1:
        raise ValueError("transformer_token_limit must be positive")
    return text[:transformer_token_limit]


def count_parts_of_speech(
    df: pd.DataFrame, model: Any, textcol: str = "sentence"
) -> pd.DataFrame:
    """Count POS tags and pronouns for each sentence.

    Parameters
    ----------
    df
        Sentence-level frame with ``id_sentence``.
    model
        spaCy model.
    textcol
        Text column to process.

    Returns
    -------
    pd.DataFrame
        POS and pronoun counts keyed by ``id_sentence``.
    """

    import spacy

    coarse_counts: list[dict[str, int]] = []
    ids: list[object] = []
    pronouns_1: list[list[str]] = []
    pronouns_2: list[list[str]] = []
    pronouns_3: list[list[str]] = []
    docs = model.pipe(df[textcol])
    for i_doc, doc in enumerate(docs):
        id_sentence = df.iloc[i_doc]["id_sentence"]
        ids.append(id_sentence)
        pos_count = doc.count_by(spacy.attrs.POS)
        pos_count = {model.vocab.strings[k]: v for k, v in pos_count.items()}
        coarse_counts.append(pos_count)
        first, second, third = categorize_pronouns(doc, model)
        pronouns_1.append(first)
        pronouns_2.append(second)
        pronouns_3.append(third)

    result = pd.DataFrame(coarse_counts).fillna(0)
    result["id_sentence"] = ids
    result["pronoun_first_person"] = pronouns_1
    result["pronoun_second_person"] = pronouns_2
    result["pronoun_third_person"] = pronouns_3
    result["count_pronoun_first_person"] = [len(x) for x in pronouns_1]
    result["count_pronoun_second_person"] = [len(x) for x in pronouns_2]
    result["count_pronoun_third_person"] = [len(x) for x in pronouns_3]
    return result


def matchy_matchy(doc: Any, pattern: list[dict[str, Any]], model: Any) -> list[str]:
    """Return spaCy text spans matching one pattern.

    Parameters
    ----------
    doc
        spaCy document to search.
    pattern
        spaCy matcher pattern.
    model
        spaCy model whose vocabulary should back the matcher.

    Returns
    -------
    list[str]
        Matched span text.
    """

    from spacy.matcher import Matcher

    matcher = Matcher(model.vocab)
    matcher.add("pattern", [pattern])
    matches = matcher(doc, as_spans=True)
    return [match.text for match in matches]


def categorize_pronouns(doc: Any, model: Any) -> tuple[list[str], list[str], list[str]]:
    """Categorize first-, second-, and third-person pronouns in a document.

    Parameters
    ----------
    doc
        spaCy document to inspect.
    model
        spaCy model whose vocabulary should back the matcher.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        First-, second-, and third-person pronoun matches.
    """

    first = matchy_matchy(
        doc,
        [{"POS": "PRON", "OP": "+", "MORPH": {"IS_SUPERSET": ["Person=1"]}}],
        model,
    )
    second = matchy_matchy(
        doc,
        [{"POS": "PRON", "OP": "+", "MORPH": {"IS_SUPERSET": ["Person=2"]}}],
        model,
    )
    third = matchy_matchy(
        doc,
        [{"POS": "PRON", "OP": "+", "MORPH": {"IS_SUPERSET": ["Person=3"]}}],
        model,
    )
    return first, second, third


def compute_ij_cosine_similarity_before_datetime(
    edf: pd.DataFrame, tdf: pd.DataFrame, model: Any
) -> pd.DataFrame:
    """Compute author-author text similarity before each interaction.

    Parameters
    ----------
    edf
        Edge-list frame with ``source_author``, ``target_author``, and
        ``datetime`` columns.
    tdf
        Sentence/text frame with ``author``, ``sentence``, and ``datetime``.
    model
        SentenceTransformer-like model exposing ``encode``.

    Returns
    -------
    pd.DataFrame
        Copy of ``edf`` with ``ij_cosine_similarity``.
    """

    from sentence_transformers import util

    result = edf.copy()
    sim_score: list[float] = []
    for i, j, dt in zip(
        result["source_author"],
        result["target_author"],
        result["datetime"],
    ):
        i_text_to_datetime = tdf[(tdf["author"] == i) & (tdf["datetime"] < dt)]
        i_text_to_datetime = i_text_to_datetime["sentence"].str.cat(sep=" ")
        i_embedding = model.encode(i_text_to_datetime, convert_to_tensor=True)

        j_text_to_datetime = tdf[(tdf["author"] == j) & (tdf["datetime"] < dt)]
        j_text_to_datetime = j_text_to_datetime["sentence"].str.cat(sep=" ")
        j_embedding = model.encode(j_text_to_datetime, convert_to_tensor=True)

        cosine_scores = util.pytorch_cos_sim(i_embedding, j_embedding)
        sim_score.append(cosine_scores[0][0].item())
    result["ij_cosine_similarity"] = sim_score
    return result
