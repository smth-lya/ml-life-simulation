import json
from collections import defaultdict
from pathlib import Path
from typing import List
from config.paths import PROCESSED_DATA_PATH

def build_vocab(tokens: List[str], min_freq: int = 3) -> List[str]:
    """Строит словарь, исключая редкие слова."""
    word_counts = defaultdict(int)
    for token in tokens:
        word_counts[token] += 1
    return [word for word, count in word_counts.items() if count >= min_freq]

def save_vocab(vocab: List[str], file_path: Path = PROCESSED_DATA_PATH):
    """Сохраняет словарь в JSON."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f)

def load_vocab(file_path: Path = PROCESSED_DATA_PATH) -> List[str]:
    """Загружает словарь из JSON."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)