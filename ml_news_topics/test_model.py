import joblib
from colorama import Fore, Style, init
import numpy as np

init(autoreset=True)

def load_model(model_path: str):
    print(f"\n{Fore.CYAN}Загрузка модели из {Fore.MAGENTA}{model_path}...")
    try:
        model = joblib.load(model_path)
        print(f"{Fore.GREEN}Модель успешно загружена!")
        return model
    except Exception as e:
        print(f"{Fore.RED}Ошибка: {e}")
        return None

def show_confidence(pred_proba, classes, top_n=3):
    """Вывод топ-N категорий с вероятностями"""
    top_indices = np.argsort(pred_proba)[::-1][:top_n]
    print(f"{Fore.YELLOW}Топ-{top_n} предсказаний:")
    for i, idx in enumerate(top_indices):
        prob = pred_proba[idx]
        color = Fore.GREEN if prob > 0.7 else Fore.YELLOW if prob > 0.3 else Fore.RED
        print(f"  {i+1}. {classes[idx]}: {color}{prob:.2%}{Style.RESET_ALL}")

def predict_with_confidence(model, text: str):
    """Предсказание с выводом уверенности"""
    # Получаем вероятности для всех категорий
    probas = model.predict_proba([text])[0]
    pred_class = model.classes_[np.argmax(probas)]

    print(f"\n{Fore.CYAN}Текст: {Style.RESET_ALL}{text[:100]}...")
    print(f"{Fore.GREEN}Предсказанная категория: {Fore.MAGENTA}{pred_class}")
    show_confidence(probas, model.classes_)

if __name__ == "__main__":
    MODEL_PATH = "models/logreg_tfidf_model.joblib"
    model = load_model(MODEL_PATH)
    if not model:
        exit(1)

    # Тестовые примеры
    examples = [
        "Apple unveils new iPhone with revolutionary AI features",
        "Manchester United wins the Champions League",
        "Study confirms Mediterranean diet reduces heart disease risk"
    ]

    for text in examples:
        predict_with_confidence(model, text)

    # Интерактивный режим
    print(f"\n{Fore.CYAN}=== ИНТЕРАКТИВНЫЙ РЕЖИМ (введите 'exit' для выхода) ===")
    while True:
        text = input(f"{Style.RESET_ALL}Введите текст новости: ")
        if text.lower() == 'exit':
            break
        predict_with_confidence(model, text)