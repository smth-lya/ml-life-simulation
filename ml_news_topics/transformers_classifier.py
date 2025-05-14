import torch
import gc
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
from colorama import Fore, Style, init
import json

from config.paths import RAW_DATA_PATH

# Инициализация
init(autoreset=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация
CONFIG = {
    "model_name": "bert-base-uncased",
    "max_length": 64,
    "batch_size": 8,
    "epochs": 2,
    "lr": 2e-5,
    "test_size": 0.1,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts  # Теперь это обычный список, а не pandas Series
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def clean_memory():
    torch.cuda.empty_cache()
    gc.collect()

def load_data():
    """Загрузка данных с преобразованием в списки"""
    try:
        logger.info(f"{Fore.CYAN}Загрузка данных...")

        # Чтение JSON напрямую в список словарей
        with open(RAW_DATA_PATH, "r") as f:
            data = [json.loads(line) for line in f]

        # Преобразуем в списки, минуя pandas
        texts = []
        categories = []

        for item in data[:20000]:  # Берем только первые 20k записей для теста
            text = item["headline"] + " " + item["short_description"]
            texts.append(text)
            categories.append(item["category"])

        # Создаем mapping категорий
        unique_cats = list(set(categories))
        cat_to_id = {cat: i for i, cat in enumerate(unique_cats)}
        labels = [cat_to_id[cat] for cat in categories]

        return train_test_split(
            texts, labels,
            test_size=CONFIG["test_size"],
            random_state=42,
            stratify=labels
        )
    except Exception as e:
        logger.error(f"{Fore.RED}Ошибка загрузки данных: {e}")
        raise

def train_model():
    try:
        # 1. Загрузка данных
        X_train, X_test, y_train, y_test = load_data()

        # 2. Инициализация модели
        logger.info(f"{Fore.YELLOW}Загрузка модели {CONFIG['model_name']}...")
        tokenizer = BertTokenizer.from_pretrained(CONFIG["model_name"])
        model = BertForSequenceClassification.from_pretrained(
            CONFIG["model_name"],
            num_labels=len(set(y_train))
        ).to(CONFIG["device"])

        # 3. Подготовка DataLoader
        train_dataset = NewsDataset(X_train, y_train, tokenizer, CONFIG["max_length"])
        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            pin_memory=True
        )

        # 4. Оптимизатор и шедулер
        optimizer = AdamW(model.parameters(), lr=CONFIG["lr"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * CONFIG["epochs"]
        )

        # 5. Цикл обучения
        logger.info(f"{Fore.GREEN}Начало обучения...")
        for epoch in range(CONFIG["epochs"]):
            model.train()
            total_loss = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for batch in progress_bar:
                optimizer.zero_grad()

                inputs = {
                    "input_ids": batch["input_ids"].to(CONFIG["device"]),
                    "attention_mask": batch["attention_mask"].to(CONFIG["device"]),
                    "labels": batch["labels"].to(CONFIG["device"])
                }

                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                progress_bar.set_postfix({"loss": total_loss/(progress_bar.n+1)})

                clean_memory()

            avg_loss = total_loss / len(train_loader)
            logger.info(f"{Fore.CYAN}Epoch {epoch+1} | Loss: {avg_loss:.4f}")

        # 6. Сохранение модели
        model.save_pretrained("bert_news_classifier")
        tokenizer.save_pretrained("bert_news_classifier")
        logger.info(f"{Fore.GREEN}Модель сохранена!")

    except Exception as e:
        logger.error(f"{Fore.RED}Критическая ошибка: {e}")
        clean_memory()
        raise

if __name__ == "__main__":
    train_model()