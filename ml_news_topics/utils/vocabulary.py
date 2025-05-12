import json
from pathlib import Path
from typing import Dict, List


def build_vocab(tokens: List[str]) -> Dict[str, int]:
    """Создает словарь токенов."""
    unique_tokens = sorted(set(tokens))
    return {token: idx for idx, token in enumerate(unique_tokens)}


def save_vocab(vocab: Dict[str, int], output_path: Path):
    """Сохраняет словарь в файл."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(input_path: Path) -> Dict[str, int]:
    """Загружает словарь из файла."""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)