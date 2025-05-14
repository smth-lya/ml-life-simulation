import json
import random
import time
from typing import Dict, List, Tuple

import torch
from colorama import Fore, Style, init
from tqdm import tqdm
from NaiveBayesClassifier import NaiveBayesClassifier
from config.paths import RAW_DATA_PATH, PROCESSED_DATA_PATH
from data.loader import load_articles
from utils.text_processor import process_text
from utils.vocabulary import build_vocab, save_vocab

# Инициализация colorama
init(autoreset=True)

def print_header():
    """Вывод красивого заголовка программы."""
    print(Fore.CYAN + "=" * 70)
    print(Fore.YELLOW + "🚀 НОВОСТНОЙ КЛАССИФИКАТОР (НАИВНЫЙ БАЙЕС)".center(70))
    print(Fore.CYAN + "=" * 70 + Style.RESET_ALL)

def print_step(step: str, description: str):
    """Вывод информации о текущем шаге."""
    print(Fore.GREEN + f"\n[{step}] {description}" + Style.RESET_ALL)
    time.sleep(0.2)

def load_and_split_data(test_ratio: float = 0.3, random_seed: int = 42) -> Tuple[List, List]:
    """Загрузка и разделение данных на train/test."""
    print_step("1", "Загрузка и подготовка данных...")

    # Загрузка данных
    articles = load_articles(RAW_DATA_PATH)
    print(Fore.LIGHTBLUE_EX + f"✔ Загружено {len(articles)} статей" + Style.RESET_ALL)

    # Разделение данных
    random.seed(random_seed)
    random.shuffle(articles)
    split_idx = int(len(articles) * (1 - test_ratio))
    train_articles = articles[:split_idx]
    test_articles = articles[split_idx:]

    print(Fore.LIGHTBLUE_EX +
          f"✔ Обучающая выборка: {len(train_articles)} статей\n"
          f"✔ Тестовая выборка: {len(test_articles)} статей" +
          Style.RESET_ALL)

    return train_articles, test_articles

def prepare_vocabulary(train_articles: List, vocab_path: str = PROCESSED_DATA_PATH) -> List:
    """Подготовка или загрузка словаря."""
    print_step("2", "Подготовка словаря...")

    if vocab_path.exists():
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        print(Fore.LIGHTBLUE_EX + f"✔ Загружен сохраненный словарь ({len(vocab)} токенов)" + Style.RESET_ALL)
    else:
        all_tokens = []
        for article in tqdm(train_articles,
                            desc="Обработка текстов",
                            unit="статья",
                            colour='green'):
            text = article.get("headline", "") + " " + article.get("short_description", "")
            tokens = process_text(text)
            all_tokens.extend(tokens)

        vocab = build_vocab(all_tokens)
        save_vocab(vocab, vocab_path)
        print(Fore.LIGHTBLUE_EX + f"✔ Создан новый словарь ({len(vocab)} токенов)" + Style.RESET_ALL)

    return vocab

def train_model(train_articles: List, vocab: List) -> NaiveBayesClassifier:
    """Обучение модели Naive Bayes."""
    print_step("3", "Обучение модели Naive Bayes...")

    classifier = NaiveBayesClassifier(vocab, alpha=1.0)
    classifier.train(train_articles)

    print(Fore.LIGHTBLUE_EX +
          f"✔ Модель обучена на {len(train_articles)} статьях\n"
          f"✔ Количество классов: {len(classifier.classes_)}" +
          Style.RESET_ALL)

    return classifier

def evaluate_model(classifier: NaiveBayesClassifier, test_articles: List) -> Dict:
    """Оценка модели на тестовых данных."""
    print_step("4", "Оценка модели на тестовых данных...")

    metrics = classifier.evaluate(test_articles)
    print_evaluation_report(metrics)

    return metrics

def print_evaluation_report(metrics: Dict):
    """Красивый вывод результатов оценки."""
    print(Fore.CYAN + "\n" + "=" * 70)
    print(Fore.YELLOW + "📊 РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ".center(70))
    print(Fore.CYAN + "=" * 70)

    # Общие метрики
    print(Fore.LIGHTGREEN_EX +
          f"\nОбщая точность: {metrics['accuracy']:.2%}\n" +
          Style.RESET_ALL)

    # Заголовок таблицы
    print(Fore.LIGHTWHITE_EX +
          f"{'Категория':<25} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'Support':<10}")
    print("-" * 65)

    # Данные по категориям
    for cat in sorted(metrics['class_metrics'].keys()):
        m = metrics['class_metrics'][cat]
        print(f"{Fore.WHITE}{cat[:24]:<25} "
              f"{Fore.LIGHTBLUE_EX}{m['precision']:<10.2f} "
              f"{m['recall']:<10.2f} "
              f"{m['f1']:<10.2f} "
              f"{Fore.LIGHTMAGENTA_EX}{m['support']:<10}")

def demo_predictions(classifier: NaiveBayesClassifier, texts: List[str] = None):
    """Демонстрация предсказаний модели."""
    print_step("5", "Демонстрация работы модели...")

    if not texts:
        texts = [
            "I love apple",
            "I love Apple",
            "Cleaner Was Dead In Belk Bathroom For 4 Days Before Body Found",
            "Stock market reaches all-time high amid economic recovery",
            "New study shows benefits of Mediterranean diet for heart health",
            "President announces new climate change initiative"
        ]

    print(Fore.CYAN + "\n" + "=" * 70)
    print(Fore.YELLOW + "📰 ДЕМО-ПРЕДСКАЗАНИЯ МОДЕЛИ".center(70))
    print(Fore.CYAN + "=" * 70)

    for text in texts:
        probs = classifier.predict_proba(text)
        top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]

        print(Fore.LIGHTWHITE_EX + f"\nТекст: {text}")
        print(Fore.LIGHTGREEN_EX + f"▶ Предсказанная категория: {top3[0][0]} (вероятность: {top3[0][1]:.2%})")

        print(Fore.LIGHTBLUE_EX + "\nТоп-3 категории:")
        for cat, prob in top3:
            print(f"  • {cat}: {prob:.2%}")

def run_pipeline(test_ratio: float = 0.3, random_seed: int = 42):
    """Запуск всего пайплайна классификации."""
    try:
        print_header()

        # 1. Загрузка и разделение данных
        train_articles, test_articles = load_and_split_data(test_ratio, random_seed)

        # 2. Подготовка словаря
        vocab = prepare_vocabulary(train_articles)

        # 3. Обучение модели
        classifier = train_model(train_articles, vocab)

        # 4. Оценка модели
        evaluate_model(classifier, test_articles)

        # 5. Демонстрация работы
        demo_predictions(classifier)

    except Exception as e:
        print(Fore.RED + f"\n⚠ Ошибка при выполнении: {str(e)}" + Style.RESET_ALL)
        raise

if __name__ == '__main__':
    print(torch.cuda.is_available())