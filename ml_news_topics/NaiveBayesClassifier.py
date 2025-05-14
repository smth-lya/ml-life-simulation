import math
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Union
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from utils.text_processor import text_to_vector  # Предполагается, что это существует

class NaiveBayesClassifier:
    def __init__(self, vocab: List[str], alpha: float = 1.0):
        """
        Инициализация классификатора.

        Args:
            vocab: Список уникальных слов в словаре
            alpha: Параметр аддитивного сглаживания (Лапласа)
        """
        self.vocab = vocab
        self.alpha = alpha
        self.category_counts = defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.total_words_per_category = defaultdict(int)
        self.classes_ = None

    def train(self, articles: List[Dict[str, Union[str, Dict]]]) -> None:
        """
        Обучение модели на статьях.

        Args:
            articles: Список статей, каждая статья - словарь с ключами:
                      - "category": категория статьи
                      - "short_description": текст для обработки
        """
        # Сброс предыдущего обучения
        self.category_counts = defaultdict(int)
        self.word_counts = defaultdict(lambda: defaultdict(int))
        self.total_words_per_category = defaultdict(int)

        # Подсчет статистик
        for article in articles:
            category = article["category"]
            self.category_counts[category] += 1

            text_vector = text_to_vector(article["short_description"], self.vocab)
            for word_idx, count in text_vector.items():
                self.word_counts[category][word_idx] += count
                self.total_words_per_category[category] += count

        # Сохраняем список уникальных классов
        self.classes_ = list(self.category_counts.keys())

    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Предсказание вероятностей для каждой категории с использованием логарифмов.

        Args:
            text: Текст для классификации

        Returns:
            Словарь {категория: вероятность}
        """
        text_vector = text_to_vector(text, self.vocab)
        log_probs = {}
        total_docs = sum(self.category_counts.values())

        for category in self.category_counts:
            # Логарифм априорной вероятности
            log_prob = math.log(self.category_counts[category] / total_docs)

            # Сумма логарифмов условных вероятностей
            for word_idx, count in text_vector.items():
                word_prob = (self.word_counts[category].get(word_idx, 0) + self.alpha) / \
                            (self.total_words_per_category[category] + self.alpha * len(self.vocab))
                log_prob += count * math.log(word_prob)

            log_probs[category] = log_prob

        # Преобразование логарифмированных вероятностей в нормальные
        max_log_prob = max(log_probs.values())
        probs = {
            cat: math.exp(log_prob - max_log_prob)  # Для численной стабильности
            for cat, log_prob in log_probs.items()
        }

        # Нормализация
        prob_sum = sum(probs.values())
        return {cat: prob / prob_sum for cat, prob in probs.items()}

    def predict(self, text: str) -> str:
        """
        Предсказание наиболее вероятной категории.

        Args:
            text: Текст для классификации

        Returns:
            Предсказанная категория
        """
        probs = self.predict_proba(text)
        return max(probs.items(), key=lambda x: x[1])[0]

    def evaluate(self, test_articles: List[Dict[str, Union[str, Dict]]]) -> Dict:
        """
        Полная оценка модели на тестовых данных с различными метриками.

        Args:
            test_articles: Список статей для тестирования

        Returns:
            Словарь с метриками:
            - accuracy: общая точность
            - class_metrics: метрики по классам
            - confusion_matrix: матрица ошибок
            - classification_report: отчет sklearn
        """
        y_true = []
        y_pred = []

        for article in test_articles:
            y_true.append(article["category"])
            y_pred.append(self.predict(article["short_description"]))

        # Общие метрики
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))

        # Метрики по классам
        class_report = classification_report(y_true, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred, labels=self.classes_)

        # Подробные метрики по классам
        class_metrics = {}
        for cat in self.classes_:
            class_metrics[cat] = {
                'precision': class_report[cat]['precision'],
                'recall': class_report[cat]['recall'],
                'f1': class_report[cat]['f1-score'],
                'support': class_report[cat]['support']
            }

        return {
            'accuracy': accuracy,
            'class_metrics': class_metrics,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report
        }

    def cross_validate(self, articles: List[Dict], n_folds: int = 5) -> Dict:
        """
        Кросс-валидация модели.

        Args:
            articles: Все статьи для кросс-валидации
            n_folds: Количество фолдов

        Returns:
            Средние метрики по всем фолдам
        """
        random.shuffle(articles)
        fold_size = len(articles) // n_folds
        metrics = []

        for i in range(n_folds):
            start = i * fold_size
            end = (i + 1) * fold_size if i < n_folds - 1 else len(articles)

            test_data = articles[start:end]
            train_data = articles[:start] + articles[end:]

            self.train(train_data)
            fold_metrics = self.evaluate(test_data)
            metrics.append(fold_metrics)

        # Усреднение метрик
        avg_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in metrics]),
            'class_metrics': {
                cat: {
                    'precision': np.mean([m['class_metrics'][cat]['precision'] for m in metrics]),
                    'recall': np.mean([m['class_metrics'][cat]['recall'] for m in metrics]),
                    'f1': np.mean([m['class_metrics'][cat]['f1'] for m in metrics]),
                    'support': metrics[0]['class_metrics'][cat]['support']  # Support одинаковый для всех фолдов
                }
                for cat in self.classes_
            }
        }

        return avg_metrics