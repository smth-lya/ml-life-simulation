from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'

RAW_DATA_PATH = DATA_DIR / 'raw' / 'News_Category_Dataset_v3.json'
PROCESSED_DATA_PATH = DATA_DIR / 'processed' / 'vocabulary.json'
NLTK_DATA_PATH = PROJECT_ROOT / 'nltk_data'