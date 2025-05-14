import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_class_weight
import joblib
from pathlib import Path
import numpy as np
from datetime import datetime
import logging
from tqdm import tqdm
from colorama import Fore, Style, init

from config.paths import RAW_DATA_PATH

# Инициализация Colorama
init(autoreset=True)

# Кастомные цвета для логов
class LogColors:
    INFO = Fore.CYAN
    WARNING = Fore.YELLOW
    ERROR = Fore.RED
    SUCCESS = Fore.GREEN
    HIGHLIGHT = Fore.MAGENTA

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format=f"{LogColors.INFO}%(asctime)s - %(levelname)s - {Style.RESET_ALL}%(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Конфигурация
CONFIG = {
    "data_path": RAW_DATA_PATH,
    "test_size": 0.2,
    "random_state": 42,
    "max_features": 10000,
    "min_df": 5,
    "ngram_range": (1, 2),
    "model_save_path": "models",
    "target_column": "category",
    "text_columns": ["headline", "short_description"]
}

def log_header(message: str) -> None:
    """Красивый заголовок с рамкой"""
    border = LogColors.HIGHLIGHT + "=" * (len(message) + 4)
    print(f"\n{border}")
    print(f"{LogColors.HIGHLIGHT}  {message}  ")
    print(f"{border}{Style.RESET_ALL}\n")

def load_data(file_path: str) -> pd.DataFrame:
    """Загрузка данных с цветным прогресс-баром"""
    logger.info(f"{LogColors.INFO}Загрузка данных из {LogColors.HIGHLIGHT}{file_path}...")
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in tqdm(f,
                                                  desc=f"{LogColors.INFO}Чтение строк",
                                                  bar_format="{l_bar}%s{bar}%s{r_bar}" % (LogColors.SUCCESS, Style.RESET_ALL))]
    logger.info(f"{LogColors.SUCCESS}Загружено {LogColors.HIGHLIGHT}{len(data)} записей")
    return pd.DataFrame(data)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Предобработка с цветными логами"""
    logger.info(f"{LogColors.INFO}Начата предобработка данных...")

    # Объединение текстовых полей
    df["text"] = df[CONFIG["text_columns"]].apply(lambda x: " ".join(x), axis=1)
    logger.info(f"{LogColors.SUCCESS}Текстовые поля объединены")

    # Очистка текста
    df["text"] = df["text"].str.lower().replace(r"[^\w\s]", "", regex=True)
    logger.info(f"{LogColors.SUCCESS}Текст очищен (нижний регистр + удалена пунктуация)")

    # Удаление пустых строк
    initial_size = len(df)
    df = df.dropna(subset=["text", CONFIG["target_column"]])
    logger.info(f"{LogColors.WARNING}Удалено {initial_size - len(df)} пустых записей")

    return df

def train_model(X_train: pd.Series, y_train: pd.Series) -> tuple:
    """Обучение модели с цветным выводом"""
    logger.info(f"{LogColors.INFO}Подсчёт весов классов...")
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    # Красивый вывод весов
    for cls, weight in list(class_weights.items())[:5]:
        logger.info(f"{LogColors.HIGHLIGHT}{cls}: {weight:.2f}")
    if len(class_weights) > 5:
        logger.info(f"{LogColors.INFO}...и ещё {len(class_weights)-5} категорий")

    # Пайплайн
    logger.info(f"{LogColors.INFO}Создание пайплайна {LogColors.HIGHLIGHT}TF-IDF + LogisticRegression")
    model = make_pipeline(
        TfidfVectorizer(
            max_features=CONFIG["max_features"],
            min_df=CONFIG["min_df"],
            ngram_range=CONFIG["ngram_range"],
            stop_words="english"
        ),
        LogisticRegression(
            max_iter=500,
            class_weight=class_weights,
            random_state=CONFIG["random_state"],
            n_jobs=-1,
            verbose=1  # Добавляем вывод логов обучения
        )
    )

    logger.info(f"{LogColors.INFO}Обучение модели...")
    model.fit(X_train, y_train)
    logger.info(f"{LogColors.SUCCESS}Обучение завершено!")
    return model

def evaluate_model(model, X_test: pd.Series, y_test: pd.Series) -> None:
    """Цветной отчёт о качестве"""
    log_header("РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Цвет accuracy в зависимости от значения
    if acc > 0.6:
        acc_color = LogColors.SUCCESS
    elif acc > 0.5:
        acc_color = LogColors.HIGHLIGHT
    else:
        acc_color = LogColors.WARNING

    print(f"\n{LogColors.INFO}Accuracy: {acc_color}{acc:.3f}{Style.RESET_ALL}")

    # Красивый отчёт классификации
    report = classification_report(y_test, y_pred)
    colored_report = ""
    for line in report.split('\n'):
        if 'avg' in line:
            colored_report += LogColors.HIGHLIGHT + line + Style.RESET_ALL + '\n'
        else:
            colored_report += line + '\n'
    print(colored_report)

def save_artifacts(model, save_dir: str) -> None:
    """Сохранение с цветным подтверждением"""
    Path(save_dir).mkdir(exist_ok=True)
    model_path = f"{save_dir}/logreg_tfidf_model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"{LogColors.SUCCESS}Модель сохранена в {LogColors.HIGHLIGHT}{model_path}")

def main():
    try:
        log_header("ЗАПУСК КЛАССИФИКАЦИИ НОВОСТЕЙ")
        start_time = datetime.now()

        # Загрузка данных
        df = load_data(CONFIG["data_path"])

        # Предобработка
        df = preprocess_data(df)
        logger.info(f"{LogColors.INFO}Финальный размер датасета: {LogColors.HIGHLIGHT}{len(df)} записей")

        # Разделение данных
        logger.info(f"{LogColors.INFO}Разделение данных на train/test...")
        X_train, X_test, y_train, y_test = train_test_split(
            df["text"],
            df[CONFIG["target_column"]],
            test_size=CONFIG["test_size"],
            random_state=CONFIG["random_state"],
            stratify=df[CONFIG["target_column"]]
        )
        logger.info(f"{LogColors.SUCCESS}Train: {LogColors.HIGHLIGHT}{len(X_train)} {LogColors.SUCCESS}записей | "
                    f"Test: {LogColors.HIGHLIGHT}{len(X_test)} {LogColors.SUCCESS}записей")

        # Обучение и оценка
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)

        # Сохранение
        save_artifacts(model, CONFIG["model_save_path"])

        # Время выполнения
        end_time = datetime.now()
        logger.info(f"{LogColors.INFO}Общее время выполнения: {LogColors.HIGHLIGHT}{end_time - start_time}")

    except Exception as e:
        logger.error(f"{LogColors.ERROR}Ошибка: {e}", exc_info=True)

if __name__ == "__main__":
    main()