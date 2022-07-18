"""Name - function dictionary to facilitate reproducible preprocessing steps."""
from src.dataset.utils import (lemmatize, remove_punctation,
                               remove_single_chars, remove_stopwords,
                               remove_whitespace, stem, word_tokenize)

preprocess_helper = {
    "whitespace": remove_whitespace,
    "tokenize": word_tokenize,
    "stopwords": remove_stopwords,
    "punctation": remove_punctation,
    "lemmatize": lemmatize,
    "single_chars": remove_single_chars,
    "stem": stem
}
