from typing import List

import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


def remove_whitespace(text) -> str:
    """Remove whitespace from text - compatibilie with pandas apply."""
    return " ".join(text.split())


def remove_stopwords(text) -> List[int]:
    """Remove english stopwords."""
    en_stopwords = stopwords.words('english')
    result = []
    for token in text:
        if token not in en_stopwords:
            result.append(token)

    return result


def remove_single_chars(text) -> List[str]:
    result = []
    for entry in text:
        if len(entry) > 1:
            result.append(entry)

    return result


def remove_punctation(text) -> str:
    """Remove punctation from text."""
    tokenizer = RegexpTokenizer(r"\w+")
    lst = tokenizer.tokenize(' '.join(text))
    return lst


def stem(text) -> List[str]:
    """Stem from text."""
    porter = PorterStemmer()
    result = []
    for word in text:
        result.append(porter.stem(word))
    return result


def lemmatize(text) -> List[str]:
    """Lemmatize the text."""
    result = []
    wordnet = WordNetLemmatizer()
    for token, tag in pos_tag(text):
        pos = tag[0].lower()
        if pos not in ['a', 'r', 'n', 'v']:
            pos = 'n'
        result.append(wordnet.lemmatize(token, pos))

    return result


def get_docs():
    """Download nltk docs used when preprocessing."""
    nltk.download("omw-1.4")
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("wordnet")
