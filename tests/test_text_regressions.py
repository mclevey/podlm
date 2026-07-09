from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "podlm"))

from podlm import text as text_mod  # noqa: E402


class FakeSentence:
    def __init__(self, text: str) -> None:
        self.text = text

    def __len__(self) -> int:
        return len(self.text.split())


class FakeDoc:
    def __init__(self, text: str) -> None:
        self.sents = [
            FakeSentence(part.strip())
            for part in text.split(".")
            if part.strip()
        ]


class FakeSentenceModel:
    def pipe(self, texts: list[str]):
        for text in texts:
            yield FakeDoc(text)


class FakeTensor:
    def __init__(self, values: list[float]) -> None:
        self.values = np.asarray(values)

    def detach(self):
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def encode(self, text: str, **kwargs):
        self.calls.append(kwargs)
        return text


class FakeSentimentModel:
    def __call__(self, encoded):
        if encoded == "bad sentence":
            raise ValueError("model failed")
        return [[FakeTensor([0.0, 1.0, 2.0])]]


class FakeEmotionModel:
    def __call__(self, text: str):
        if text == "bad sentence":
            raise ValueError("model failed")
        return [
            [
                {"label": label, "score": float(index)}
                for index, label in enumerate(text_mod.GO_EMOTIONS_LABELS)
            ]
        ]


class FakeEntityModel:
    def predict(self, text: str):
        if text == "bad sentence":
            raise ValueError("model failed")
        return [
            {"span": "John", "label": "PERSON", "score": 0.95},
            {"span": "MUN", "label": "ORG", "score": 0.40},
        ]


class TextRegressionTests(unittest.TestCase):
    def test_split_sentences_preserves_ids_when_non_string_rows_exist(self) -> None:
        df = pd.DataFrame(
            {
                "id": ["skip-me", "keep-me"],
                "text": [np.nan, "one two three. four five six"],
            }
        )

        result = text_mod.split_sentences(df, FakeSentenceModel())

        self.assertEqual(result["id_sentence"].tolist(), ["keep-me_0", "keep-me_1"])
        self.assertEqual(result["sentence"].tolist(), ["one two three", "four five six"])

    def test_truncate_text_returns_truncated_text_without_logging(self) -> None:
        result = text_mod.truncate_text_to_transformer_limit("abcdef", 3)

        self.assertEqual(result, "abc")

    def test_transformer_sentiment_keeps_ids_and_uses_tokenizer_truncation(self) -> None:
        tokenizer = FakeTokenizer()
        df = pd.DataFrame(
            {
                "id_sentence": ["s1"],
                "sentence": ["good sentence"],
            }
        )

        result = text_mod.transformer_sentiment(
            df,
            model=FakeSentimentModel(),
            tokenizer=tokenizer,
        )

        self.assertEqual(result["id_sentence"].tolist(), ["s1"])
        self.assertTrue(tokenizer.calls[0]["truncation"])
        self.assertEqual(tokenizer.calls[0]["max_length"], 512)
        self.assertGreater(
            result.loc[0, "sentiment_positive"],
            result.loc[0, "sentiment_neutral"],
        )

    def test_transformer_sentiment_raises_instead_of_silently_dropping_rows(self) -> None:
        df = pd.DataFrame(
            {
                "id_sentence": ["s1", "s2"],
                "sentence": ["good sentence", "bad sentence"],
            }
        )

        with self.assertRaisesRegex(RuntimeError, "sentiment labeling failed"):
            text_mod.transformer_sentiment(
                df,
                model=FakeSentimentModel(),
                tokenizer=FakeTokenizer(),
            )

    def test_transformer_emotion_concepts_matches_call_site_and_keeps_alignment(self) -> None:
        df = pd.DataFrame(
            {
                "id_sentence": ["s1", "s2"],
                "sentence": ["first sentence", "second sentence"],
            }
        )

        result = text_mod.transformer_emotion_concepts(
            df,
            FakeEmotionModel(),
            "sentence",
            "id_sentence",
        )

        self.assertEqual(result["id_sentence"].tolist(), ["s1", "s2"])
        self.assertEqual(
            result.loc[0, text_mod.GO_EMOTIONS_LABELS[-1]],
            float(len(text_mod.GO_EMOTIONS_LABELS) - 1),
        )

    def test_transformer_emotion_concepts_raises_on_model_failure(self) -> None:
        df = pd.DataFrame(
            {
                "id_sentence": ["s1", "s2"],
                "sentence": ["first sentence", "bad sentence"],
            }
        )

        with self.assertRaisesRegex(RuntimeError, "emotion concept labeling failed"):
            text_mod.transformer_emotion_concepts(df, FakeEmotionModel())

    def test_transformer_entities_returns_counts_with_id_sentence_column(self) -> None:
        df = pd.DataFrame(
            {
                "id_sentence": ["s1"],
                "sentence": ["John works at MUN"],
            }
        )

        entities, counts = text_mod.transformer_entities(df, FakeEntityModel(), 0.5)

        self.assertEqual(entities["id_sentence"].tolist(), ["s1", "s1"])
        self.assertIn("id_sentence", counts.columns)
        self.assertIn("entity_count_person", counts.columns)
        self.assertNotIn("entity_count_id_sentence", counts.columns)


if __name__ == "__main__":
    unittest.main()
