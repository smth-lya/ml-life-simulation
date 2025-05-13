import random
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

    def test(self, test_articles):
        """Оценивает точность модели на тестовых данных.

        Args:
            test_articles (list): Список статей для тестирования.

        Returns:
            float: Accuracy (доля верных предсказаний).
            dict: Подробный отчет по классам.
        """
        correct = 0
        total = len(test_articles)
        class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

        for article in test_articles:
            true_category = article["category"]
            predicted = self.predict(article["short_description"])

            class_stats[true_category]['total'] += 1
            if predicted == true_category:
                correct += 1
                class_stats[true_category]['correct'] += 1

        accuracy = correct / total
        report = {
            'accuracy': accuracy,
            'class_metrics': {
                cat: {
                    'precision': stats['correct'] / sum(
                        1 for a in test_articles
                        if self.predict(a["short_description"]) == cat
                    ) if any(
                        self.predict(a["short_description"]) == cat
                        for a in test_articles
                    ) else 0,
                    'recall': stats['correct'] / stats['total'],
                    'f1': 2 * (stats['correct'] / stats['total']) / (
                        (stats['correct'] / stats['total']) +
                        (stats['correct'] / sum(
                            1 for a in test_articles
                            if self.predict(a["short_description"]) == cat
                        )) if any(
                            self.predict(a["short_description"]) == cat
                            for a in test_articles
                        ) else 1
                    ) if stats['correct'] > 0 else 0
                }
                for cat, stats in class_stats.items()
            }
        }

        return accuracy, report

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
