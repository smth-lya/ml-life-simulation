import torch
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    pipeline
)
import pandas as pd
import numpy as np
from colorama import Fore, Style, init

init(autoreset=True)

# Конфигурация
CONFIG = {
    "model_path": "bert_news_classifier",  # Папка с сохранённой моделью
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "max_length": 128
}

def load_model():
    """Загрузка модели и токенизатора"""
    print(f"{Fore.CYAN}\nЗагрузка модели из {CONFIG['model_path']}...")

    try:
        # Определяем тип модели автоматически
        if "roberta" in CONFIG["model_path"].lower():
            tokenizer = RobertaTokenizer.from_pretrained(CONFIG["model_path"])
            model = RobertaForSequenceClassification.from_pretrained(CONFIG["model_path"])
        else:
            tokenizer = BertTokenizer.from_pretrained(CONFIG["model_path"])
            model = BertForSequenceClassification.from_pretrained(CONFIG["model_path"])

        model.to(CONFIG["device"])
        print(f"{Fore.GREEN}Модель успешно загружена!")
        return model, tokenizer

    except Exception as e:
        print(f"{Fore.RED}Ошибка загрузки модели: {e}")
        return None, None

def predict(text: str, model, tokenizer, id_to_label: dict):
    """Предсказание с выводом вероятностей"""
    inputs = tokenizer(
        text,
        max_length=CONFIG["max_length"],
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(CONFIG["device"])

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    top_probs, top_indices = torch.topk(probs, 3)

    print(f"\n{Fore.YELLOW}Текст: {Style.RESET_ALL}{text}")
    print(f"{Fore.CYAN}Топ-3 предсказания:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        prob_percent = prob.item() * 100
        color = Fore.GREEN if prob_percent > 70 else Fore.YELLOW if prob_percent > 30 else Fore.RED
        print(f"  {i+1}. {id_to_label[idx.item()]}: {color}{prob_percent:.1f}%{Style.RESET_ALL}")

def interactive_mode(model, tokenizer, id_to_label):
    """Интерактивный режим"""
    print(f"\n{Fore.MAGENTA}=== ИНТЕРАКТИВНЫЙ РЕЖИМ ===")
    print(f"{Fore.CYAN}Введите текст новости (или 'exit' для выхода):")

    while True:
        text = input(f"{Style.RESET_ALL}> ")
        if text.lower() == 'exit':
            break
        predict(text, model, tokenizer, id_to_label)

def test_examples(model, tokenizer, id_to_label):
    """Тестовые примеры"""
    examples = [
        "Apple unveils new iPhone with revolutionary AI features",
        "Manchester United wins the Champions League",
        "Study confirms Mediterranean diet reduces heart disease risk",
        "I love apple",  # Проверка на омонимы
        "President signs new climate change bill"
    ]

    print(f"{Fore.BLUE}\nЗапуск тестовых примеров:")
    for example in examples:
        predict(example, model, tokenizer, id_to_label)

if __name__ == "__main__":
    # Загрузка модели
    model, tokenizer = load_model()
    if model is None:
        exit(1)

    # Загрузка соответствий id -> метка
    df = pd.read_json("News_Category_Dataset_v3.json", lines=True)
    categories = df["category"].unique()
    id_to_label = {i: cat for i, cat in enumerate(categories)}

    # Тестовые примеры
    test_examples(model, tokenizer, id_to_label)

    # Интерактивный режим
    interactive_mode(model, tokenizer, id_to_label)