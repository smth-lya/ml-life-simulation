# News Category Classifier - Документация

## 📌 О проекте

Проект представляет собой классификатор новостных статей по 42 категориям на основе алгоритма Naive Bayes. Система анализирует заголовки и краткие описания статей, чтобы определить их тематическую принадлежность.

## 🚀 Быстрый старт

### Предварительные требования
- Python 3.8+
- pip

### Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/news-classifier.git
cd news-classifier
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Загрузите модель Spacy:
```bash
python -m spacy download en_core_web_sm
```

### Запуск

1. Предобработка данных:
```bash
python scripts/preprocess_data.py
```

2. Обучение модели:
```bash
python scripts/train_model.py
```

3. Запуск классификации (пример):
```bash
python scripts/predict.py --text "New study shows benefits of Mediterranean diet"
```

## 🗂 Структура проекта

```
news-classifier/
├── config/         # Конфигурационные файлы
├── data/           # Исходные и обработанные данные
├── models/         # Обученные модели
├── notebooks/      # Исследовательские ноутбуки
├── pipeline/       # Основные этапы обработки
├── scripts/        # Исполняемые скрипты
├── tests/          # Тесты
└── utils/          # Вспомогательные утилиты
```

## ⚙️ Конфигурация

Основные параметры можно изменить в файлах:
- `config/paths.py` - пути к данным
- `config/params.py` - гиперпараметры моделей

## 🔧 Основные компоненты

### 1. Обработка данных
- **Загрузка данных**: `utils/data_loader.py`
- **Очистка текста**: `utils/text_processor.py`
- **Построение словаря**: `utils/vocabulary.py`

### 2. Модель классификации
- Реализация Naive Bayes: `pipeline/train.py`
- Альтернативные модели можно добавить в `pipeline/models/`

### 3. Исполняемые скрипты
- `preprocess_data.py` - подготовка данных
- `train_model.py` - обучение модели
- `predict.py` - классификация текста

## 📊 Данные

Используется датасет [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) с Kaggle:
- 210k новостных статей
- 42 категории
- Поля: заголовок, описание, категория, авторы, дата

Пример записи:
```json
{
  "category": "POLITICS",
  "headline": "New bill proposed in Congress",
  "short_description": "Lawmakers introduced new legislation today...",
  "authors": "John Smith",
  "date": "2022-01-01"
}
```

## 🤖 API

Для интеграции с другими сервисами можно использовать REST API:

```python
from fastapi import FastAPI
from pipeline.predict import Predictor

app = FastAPI()
predictor = Predictor.load()

@app.post("/predict")
async def predict(text: str):
    return {"category": predictor.predict(text)}
```

Запуск сервера:
```bash
uvicorn api:app --reload
```

## 🧪 Тестирование

Запуск тестов:
```bash
python -m pytest tests/
```

Покрытие тестами:
```bash
pytest --cov=. tests/
```

## 📈 Производительность

Метрики на тестовой выборке:
- Accuracy: 0.78
- F1-score (macro): 0.76
- Время предсказания: ~50ms/текст

## 📚 Дополнительные материалы

1. [Описание датасета на Kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
2. [Документация Spacy](https://spacy.io/)
3. [Теория Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)

## 🤝 Как можно помочь

1. Добавление новых моделей
2. Улучшение предобработки текста
3. Оптимизация производительности
4. Расширение тестового покрытия

## 📝 Лицензия

MIT License. Подробнее в файле LICENSE.