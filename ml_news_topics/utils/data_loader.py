import json
from pathlib import Path
from typing import List, Dict


def load_articles(file_path: Path) -> List[Dict]:
    """Загружает статьи из JSON файла."""
    articles = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                articles.append(data)
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл {file_path} не найден")
    except json.JSONDecodeError:
        raise ValueError(f"Ошибка при чтении JSON из файла {file_path}")

    return articles


def load_short_descriptions(file_path: Path) -> List[str]:
    """Загружает только short_description из JSON-файла."""
    descriptions = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'short_description' in data:
                        descriptions.append(data['short_description'])
                except json.JSONDecodeError:
                    print(f"Ошибка в строке: {line}")  # Пропускаем битые строки
                    continue
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл {file_path} не найден")

    return descriptions