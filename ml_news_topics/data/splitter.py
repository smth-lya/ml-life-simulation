import random
from typing import List, Tuple
from config.settings import TEST_RATIO, RANDOM_SEED

def split_data(articles: List, test_ratio: float = TEST_RATIO) -> Tuple[List, List]:
    """Разделяет данные на train/test."""
    random.seed(RANDOM_SEED)
    random.shuffle(articles)
    split_idx = int(len(articles) * (1 - test_ratio))
    return articles[:split_idx], articles[split_idx:]