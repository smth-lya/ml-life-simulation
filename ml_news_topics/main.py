from config.paths import RAW_DATA_PATH, PROCESSED_DATA_PATH
from utils.data_loader import load_articles
from utils.text_processor import process_text
from utils.vocabulary import build_vocab, save_vocab
from tqdm import tqdm
from colorama import init, Fore, Style
from NaiveBayesClassifier import NaiveBayesClassifier
import time
import json
import random
from collections import defaultdict

init(autoreset=True)

def split_data(articles, test_ratio=0.3, random_seed=None):
    """Разделяет данные на обучающую и тестовую выборки"""
    if random_seed is not None:
        random.seed(random_seed)

    shuffled = articles.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]

def print_header():
    print(Fore.CYAN + "=" * 50)
    print(Fore.YELLOW + "🚀 НОВОСТНОЙ КЛАССИФИКАТОР НА БАЙЕСЕ".center(50))
    print(Fore.CYAN + "=" * 50 + Style.RESET_ALL)

def print_step(step, description):
    print(Fore.GREEN + f"\n[{step}] {description}" + Style.RESET_ALL)
    time.sleep(0.3)

def evaluate_model(classifier, test_articles):
    """Оптимизированная оценка точности модели"""
    print_step("5", "Оценка точности модели...")

    correct = 0
    total = len(test_articles)
    class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    predictions = []

    # Сначала собираем все предсказания
    for article in tqdm(test_articles, desc="Предсказание", unit="статья", colour='blue'):
        predictions.append(classifier.predict(article["short_description"]))

    # Затем вычисляем метрики
    for article, predicted in zip(test_articles, predictions):
        true_category = article["category"]
        class_stats[true_category]['total'] += 1
        if predicted == true_category:
            correct += 1
            class_stats[true_category]['correct'] += 1

    accuracy = correct / total

    # Оптимизированный расчет метрик
    report = {
        'accuracy': accuracy,
        'class_metrics': {}
    }

    # Создаем словарь для быстрого доступа
    pred_counts = defaultdict(int)
    for pred in predictions:
        pred_counts[pred] += 1

    for cat, stats in class_stats.items():
        preds_for_class = pred_counts.get(cat, 0)

        precision = stats['correct'] / preds_for_class if preds_for_class > 0 else 0
        recall = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        report['class_metrics'][cat] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': stats['total']
        }

    return report

def print_report(report):
    """Красивый вывод результатов оценки"""
    print(Fore.CYAN + "\n" + "=" * 50)
    print(Fore.YELLOW + "📊 РЕЗУЛЬТАТЫ ОЦЕНКИ".center(50))
    print(Fore.CYAN + "=" * 50)

    print(Fore.LIGHTGREEN_EX + f"\nОбщая точность: {report['accuracy']:.2%}")

    print(Fore.LIGHTBLUE_EX + "\nДетали по категориям:")
    for cat, metrics in report['class_metrics'].items():
        print(f"{Fore.WHITE}{cat}: "
              f"Prec={metrics['precision']:.2f}, "
              f"Rec={metrics['recall']:.2f}, "
              f"F1={metrics['f1']:.2f} "
              f"(n={metrics['support']})")

def main():
    print_header()

    # 1. Загрузка данных
    print_step("1", "Загрузка статей...")
    articles = load_articles(RAW_DATA_PATH)
    print(Fore.LIGHTBLUE_EX + f"✔ Загружено {len(articles)} статей" + Style.RESET_ALL)

    # 2. Разделение данных
    print_step("2", "Разделение на train/test...")
    train_articles, test_articles = split_data(articles, test_ratio=0.3, random_seed=42)
    print(Fore.LIGHTBLUE_EX +
          f"✔ Обучающая выборка: {len(train_articles)} статей\n"
          f"✔ Тестовая выборка: {len(test_articles)} статей" +
          Style.RESET_ALL)

    # 3. Подготовка словаря
    if PROCESSED_DATA_PATH.exists():
        print_step("3", "Загрузка сохранённого словаря...")
        with open(PROCESSED_DATA_PATH, 'r') as f:
            vocab = json.load(f)
        print(Fore.LIGHTBLUE_EX + f"✔ Загружен словарь ({len(vocab)} токенов)" + Style.RESET_ALL)
    else:
        print_step("3", "Создание нового словаря...")
        all_tokens = []
        for article in tqdm(train_articles, desc="Обработка", unit="статья", colour='green'):
            tokens = process_text(article.get("headline", "") + " " + article.get("short_description", ""))
            all_tokens.extend(tokens)

        vocab = build_vocab(all_tokens)
        save_vocab(vocab, PROCESSED_DATA_PATH)
        print(Fore.LIGHTBLUE_EX + f"✔ Создан и сохранён новый словарь ({len(vocab)} токенов)" + Style.RESET_ALL)

    # 4. Обучение модели
    print_step("4", "Обучение модели...")
    classifier = NaiveBayesClassifier(vocab)
    classifier.train(train_articles)
    print(Fore.LIGHTBLUE_EX + f"✔ Модель обучена на {len(train_articles)} статьях" + Style.RESET_ALL)

    # 5. Оценка модели
    report = evaluate_model(classifier, test_articles)
    print_report(report)

    # 6. Демо-предсказание
    test_text = "Cleaner Was Dead In Belk Bathroom For 4 Days Before Body Found"
    predicted_category = classifier.predict(test_text)

    print(Fore.CYAN + "\n" + "=" * 50)
    print(Fore.YELLOW + f"📰 ДЕМО-ПРЕДСКАЗАНИЕ".center(50))
    print(Fore.CYAN + "=" * 50)
    print(Fore.LIGHTWHITE_EX + f"\nТекст: {test_text}")
    print(Fore.LIGHTGREEN_EX + f"Предсказанная категория: {predicted_category}" + Style.RESET_ALL)

if __name__ == '__main__':
    main()