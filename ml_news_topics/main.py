import json
from stop_words import stop_words
from nltk.stem import WordNetLemmatizer, PorterStemmer

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def load_articles_from_json(filepath):
    articles = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            articles.append(data)
    return articles

def clean_and_tokenize(text):
    text = text.lower()
    cleaned = "".join(char if char.isalpha() or char == " " else " " for char in text)
    words = cleaned.split()

    tokens = []
    for word in words:
        if word not in stop_words:
            lemma = lemmatizer.lemmatize(word)
            stemmed = stemmer.stem(lemma)
            tokens.append(stemmed)

    return tokens


articles = load_articles_from_json("../News_Category_Dataset_v3.json")
all_tokens = []

for article in articles:
    description = article.get("short_description", "")
    tokens = clean_and_tokenize(description)
    all_tokens.extend(tokens)

unique_tokens = sorted(set(all_tokens))
word_dict = {word: idx for idx, word in enumerate(unique_tokens)}

for word, idx in word_dict.items():
    print(f'"{word}": {idx},')
