import re
from typing import Dict, List

import pandas as pd


def split_sentences(text: str) -> List[str]:
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]
    return sentences or [text]


def build_parent_child_chunks(df: pd.DataFrame, window_size: int = 1) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for parent_row_index, row in df.reset_index(drop=True).iterrows():
        parent_text = str(row["claim_text"])
        sentences = split_sentences(parent_text)
        claim_id = str(row["claim_id"])

        for sentence_index, sentence in enumerate(sentences):
            start = max(0, sentence_index - window_size)
            end = min(len(sentences), sentence_index + window_size + 1)
            window_text = " ".join(sentences[start:end])
            rows.append(
                {
                    "chunk_id": f"{claim_id}::sent::{sentence_index}",
                    "parent_row_index": int(parent_row_index),
                    "claim_id": claim_id,
                    "chunk_type": "sentence_window_child",
                    "child_text": sentence,
                    "window_text": window_text,
                    "parent_text": parent_text,
                    "sentence_index": int(sentence_index),
                    "window_start": int(start),
                    "window_end": int(end - 1),
                }
            )

        rows.append(
            {
                "chunk_id": f"{claim_id}::parent",
                "parent_row_index": int(parent_row_index),
                "claim_id": claim_id,
                "chunk_type": "parent_claim",
                "child_text": parent_text,
                "window_text": parent_text,
                "parent_text": parent_text,
                "sentence_index": -1,
                "window_start": 0,
                "window_end": max(0, len(sentences) - 1),
            }
        )

    return pd.DataFrame(rows)
