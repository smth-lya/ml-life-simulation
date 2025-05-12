from collections import defaultdict
from utils.text_processor import text_to_vector
import math


class NaiveBayesClassifier:
    def __init__(self, vocab):
        self.vocab = vocab
        self.category_counts = defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.total_words_per_category = defaultdict(int)

    def train(self, articles):
        """Обучение модели на статьях."""
        for article in articles:
            category = article["category"]
            self.category_counts[category] += 1
            text_vector = text_to_vector(article["short_description"], self.vocab)
            for word_idx, count in text_vector.items():
                self.word_counts[category][word_idx] += count
                self.total_words_per_category[category] += count

    def predict(self, text):
        """Предсказание категории для текста (без логарифмов)."""
        text_vector = text_to_vector(text, self.vocab)
        best_category = None
        max_prob = 0.0  # Максимальная вероятность

        for category in self.category_counts:
            # Априорная вероятность категории
            prob = self.category_counts[category] / sum(self.category_counts.values())

            # Произведение условных вероятностей слов
            for word_idx, count in text_vector.items():
                word_prob = (self.word_counts[category].get(word_idx, 0) + 1) / \
                            (self.total_words_per_category[category] + len(self.vocab))
                prob *= (word_prob ** count)

            if prob > max_prob:
                max_prob = prob
                best_category = category

        return best_category
