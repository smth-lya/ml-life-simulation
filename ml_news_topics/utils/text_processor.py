# from stop_words import stop_words
from typing import List
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from typing import List
from collections import defaultdict

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def process_text(text: str) -> List[str]:
    """Очищает и токенизирует текст"""
    if not text or not isinstance(text, str):
        return []

    text = text.lower()
    # Удаляем все, кроме букв и пробелов
    cleaned = ''.join(c if c.isalpha() or c == ' ' else ' ' for c in text)
    words = cleaned.split()

    tokens = []
    for word in words:
        if word not in stop_words and len(word) > 2:  # Проверка против NLTK стоп-слов
            lemma = lemmatizer.lemmatize(word)
            stem = stemmer.stem(lemma)
            tokens.append(stem)

    return tokens

def text_to_vector(text, vocab):
    """Преобразование текста в вектор (частоты слов по BoW)."""
    vector = defaultdict(int)
    tokens = process_text(text)  # Ваша функция обработки текста
    for word in tokens:
        if word in vocab:
            vector[vocab[word]] += 1  # Используем индекс слова из vocab
    return vector


"""Времеенно выборки на которых модель учится и на которых будет проверяться совпадают"""
def split_data(articles):
    return articles.copy(), articles.copy()
