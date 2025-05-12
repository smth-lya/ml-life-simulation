from config.paths import RAW_DATA_PATH, PROCESSED_DATA_PATH
from utils.data_loader import load_articles
from utils.text_processor import process_text, split_data
from utils.vocabulary import build_vocab, save_vocab
from tqdm import tqdm
from colorama import init, Fore, Style
from NaiveBayesClassifier import NaiveBayesClassifier
import time
import json

init(autoreset=True)


def print_header():
    print(Fore.CYAN + "=" * 50)
    print(Fore.YELLOW + "🚀 ОБРАБОТКА НОВОСТНЫХ СТАТЕЙ".center(50))
    print(Fore.CYAN + "=" * 50 + Style.RESET_ALL)


def print_step(step, description):
    print(Fore.GREEN + f"\n[{step}] {description}" + Style.RESET_ALL)
    time.sleep(0.3)


def main():
    print_header()

    # Загрузка статей
    print_step("1", "Загрузка статей...")
    articles = load_articles(RAW_DATA_PATH)
    print(Fore.LIGHTBLUE_EX + f"✔ Загружено {len(articles)} статей" + Style.RESET_ALL)

    # Проверяем, существует ли сохранённый словарь
    if PROCESSED_DATA_PATH.exists():
        print_step("2", "Загрузка сохранённого словаря...")
        with open(PROCESSED_DATA_PATH, 'r') as f:
            vocab = json.load(f)
        print(Fore.LIGHTBLUE_EX + f"✔ Загружен словарь ({len(vocab)} токенов)" + Style.RESET_ALL)
    else:
        print_step("2", "Создание нового словаря...")
        all_tokens = []
        for article in tqdm(articles, desc="Обработка", unit="статья", colour='green'):
            tokens = process_text(article.get("headline", ""))
            all_tokens.extend(tokens)

        vocab = build_vocab(all_tokens)

        # Сохраняем словарь для будущего использования
        with open(PROCESSED_DATA_PATH, 'w') as f:
            json.dump(vocab, f)
        print(Fore.LIGHTBLUE_EX + f"✔ Создан и сохранён новый словарь ({len(vocab)} токенов)" + Style.RESET_ALL)

    # Обучение модели
    print_step("3", "Обучение наивного Байеса...")
    classifier = NaiveBayesClassifier(vocab)
    classifier.train(articles)

    # Предсказание для конкретного текста
    test_text = "Cleaner Was Dead In Belk Bathroom For 4 Days Before Body Found"
    predicted_category = classifier.predict(test_text)

    print(Fore.CYAN + "\n" + "=" * 50)
    print(Fore.YELLOW + f"Предсказанная категория: {predicted_category}".center(50))
    print(Fore.CYAN + "=" * 50 + Style.RESET_ALL)


if __name__ == '__main__':
    main()