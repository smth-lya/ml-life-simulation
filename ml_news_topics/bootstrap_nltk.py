import nltk

def download_nltk_resources():
    """Загружает необходимые ресурсы NLTK."""
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('stopwords')

if __name__ == '__main__':
    download_nltk_resources()