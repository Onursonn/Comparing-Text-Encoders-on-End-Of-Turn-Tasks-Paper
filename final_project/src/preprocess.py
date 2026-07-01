"""FTAD parsing and text normalization (raw vs normalized)."""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
TOKEN_PATTERN = re.compile(r"[a-zA-Z]+|[0-9]+|[^\w\s]")


@dataclass(frozen=True)
class EOTExample:
    context: str
    utterance: str
    label: int


def normalize(text: str, mode: str = "normalized") -> str:
    if mode == "raw":
        return text.strip()
    if mode != "normalized":
        raise ValueError(f"Unknown mode: {mode}")
    text = text.lower().translate(_PUNCT_TABLE)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def parse_ftad_line(line: str) -> EOTExample:
    parts = line.rstrip("\n").split("\t")
    if len(parts) != 3:
        raise ValueError(f"Expected 3 tab-separated fields, got {len(parts)}")
    context, utterance, label_str = parts
    return EOTExample(context=context, utterance=utterance, label=int(label_str))


def load_ftad_split(path) -> list[EOTExample]:
    examples: list[EOTExample] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(parse_ftad_line(line))
    return examples


def build_input_text(context: str, utterance: str, use_context: bool) -> str:
    if use_context and context.strip():
        context_text = context.replace("|", " ")
        return f"{context_text} {utterance}".strip()
    return utterance.strip()


def prepare_text(
    example: EOTExample,
    *,
    text_mode: str,
    use_context: bool,
) -> str:
    raw = build_input_text(example.context, example.utterance, use_context)
    return normalize(raw, mode=text_mode)


def pad_sequence(sequence: list[int], max_length: int, pad_id: int = 0) -> list[int]:
    sequence = sequence[:max_length]
    return sequence + [pad_id] * (max_length - len(sequence))


class Vocabulary:
    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {idx: token for token, idx in token_to_id.items()}

    @classmethod
    def build(
        cls,
        texts: list[str],
        *,
        min_freq: int = 1,
    ) -> Vocabulary:
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tokenize(text))

        token_to_id = {cls.PAD: 0, cls.UNK: 1}
        for token, count in sorted(counts.items()):
            if count >= min_freq:
                token_to_id[token] = len(token_to_id)
        return cls(token_to_id)

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> list[int]:
        unk = self.token_to_id[self.UNK]
        return [self.token_to_id.get(token, unk) for token in tokenize(text)]

    def binary_bow(self, text: str) -> list[float]:
        vector = [0.0] * len(self)
        for token_id in self.encode(text):
            vector[token_id] = 1.0
        return vector
