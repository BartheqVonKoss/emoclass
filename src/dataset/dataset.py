import joblib
from src.dataset.preprocess_helper import preprocess_helper
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize,pos_tag
from nltk import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from typing import List, Union, Dict, Optional
import json
from pathlib import Path


class GoEmotionsDataset:
    def __init__(self, labels_file: Optional[Union[Path, str]], emotions_path: Union[Path, str], limit: Optional[int]=None, train_cfg=None):
        # self.data_file = data_file
        self.labels: List = None
        self.features = None
        self.train_cfg = train_cfg
        self.data: List = []
        self.limit = limit
        self.names_dict: Dict[str, int] = None
        with open(emotions_path, encoding="UTF-8") as emotions_file:
            self.emotions = [emotion.split("\n")[0] for emotion in emotions_file.readlines()]

    # def get_labels(self):
    #     if labels_file:
    #         with open(labels_file, encoding="UTF-8") as labels_fil:
    #             self.label_mapping = json.load(labels_fil)
    #     else:
    #         self.label_mapping = 

    def __len__(self):
        return len(self.data)

    def load_tsv(self, tsv_path):
        """Load tsv file into a dataframe."""
        self.data = pd.read_csv(tsv_path, sep="\t", names=["text", "label", "id"])
        if self.limit is not None:
            self.data = self.data.iloc[:self.limit]
        # the code below outputs strange characters, let's go with pandas
        # with open(tsv_path, encoding="UTF-8") as tsv_file:
        #     self.names_dict = {"text": 0, "label": 1, "id": 2}  # to access easier
        #     for line in tsv_file:
        #         self.data.append(line.strip("\n").split(","))



    def load_data(self, data_path: Union[Path, str, List[Union[Path, str]]]):
        """Method to support multiple datafiles to be used."""
        if isinstance(self.index, Path or str):
            self.load_tsv(self.index)
        elif isinstance(self.index, list):
            for idx in self.index:
                self.load_tsv(idx)


    def initial_preprocess(self):
        """Apply sequentially cleaning tools from nltk to data."""
        # TODO wrap it in some sequential opeartion, like for over dict of options
        # print(self.data.head())
        self.data["proc"] = self.data.text.copy()
        for n, func in preprocess_helper.items():
            self.data.proc = self.data.proc.apply(lambda x: func(x))
        # self.data.proc = self.data.proc.str.lower()
        
        # # print(self.data.head())
        # self.data.proc = self.data.proc.apply(lambda x: remove_whitespace(x))
        # # print(self.data.head())
        # self.data.proc = self.data.proc.apply(lambda x: word_tokenize(x))
        # # print(self.data.head())
        # # consider spell checking and correction
        # self.data.proc = self.data.proc.apply(lambda x: remove_stopwords(x))
        # # print(self.data.head())
        # self.data.proc = self.data.proc.apply(lambda x: remove_punctation(x))
        # # print(self.data.head())
        # # frequent_list = get_frequent_words(self.data)
        # # self.data.proc = self.data.proc.apply(lambda x: remove_freq_words(x, frequent_list))
        # # print(self.data.head())
        # self.data.proc = self.data.proc.apply(lambda x: lemmatize(x))
        # # print(self.data.head())
        # self.data.proc = self.data.proc.apply(lambda x: remove_single_chars(x))
        # # print(self.data.head())
        # self.data.proc = self.data.proc.apply(lambda x: stem(x))
        # print(self.data.head())


    def extract_features(self, transform: bool=False):
        """Extract features from preprocessed dataset."""
        # bag of words
        # vectorizer = CountVectorizer(ngram_range=(1, 2))
        self.data["proc_t"] = self.data.proc.apply(lambda x: " ".join(x))
        self.data["labels"] = self.data.label.apply(lambda x: int(x.split(',')[0]))
        if not transform:
            # this is case for training dataset
            vectorizer = CountVectorizer()
            features = vectorizer.fit_transform(self.data.proc_t).toarray()
            # dump vocabulary to be used later
            with open(Path(self.train_cfg.CHECKPOINTS_DIR / "features.joblib"), "wb") as buf:
                joblib.dump(vectorizer.vocabulary_, buf)
            # pickle.dump(vectorizer.vocabulary_, open(Path(self.train_cfg.CHECKPOINTS_DIR / "feature.pkl"), "wb"))
        else:
            with open(Path(self.train_cfg.CHECKPOINTS_DIR / "features.joblib"), "rb") as buf:
                vectorizer = CountVectorizer(vocabulary=pickle.load(buf))
                features = vectorizer.transform(self.data.proc_t).toarray()

        dataset = {"features": features, "labels": np.asarray(self.data.labels)}

        return dataset


def remove_whitespace(text):
    """Remove whitespace from text - compatibilie with pandas apply."""
    return  " ".join(text.split())

def remove_stopwords(text):
    """Remove english stopwords."""
    en_stopwords = stopwords.words('english')
    result = []
    for token in text:
        if token not in en_stopwords:
            result.append(token)

    return result

def remove_single_chars(text):
    result = []
    for entry in text:
        if len(entry) > 1:
            result.append(entry)

    return result

def remove_punctation(text):
    """Remove punctation from text."""
    tokenizer = RegexpTokenizer(r"\w+")
    lst = tokenizer.tokenize(' '.join(text))
    return lst

def stem(text):
    porter = PorterStemmer()
    
    result=[]
    for word in text:
        result.append(porter.stem(word))
    return result

def lemmatize(text):
    result = []
    wordnet = WordNetLemmatizer()
    for token, tag in pos_tag(text):
        pos = tag[0].lower()
        
        if pos not in ['a', 'r', 'n', 'v']:
            pos = 'n'
            
        result.append(wordnet.lemmatize(token,pos))
    
    return result
# def get_frequent_words(dataframe):
#     lst = []
#     import numpy as np
#     lst = [np.array( text ) for text in dataframe.text]
#     lst = np.array(lst).flatten()
#     print(lst)
#     exit()
#     fdist = FreqDist(lst)
#     return fdist.most_common(10)

# def remove_freq_words(text, frequent_list):
#     result = []
#     for item in text:
#         if item not in frequent_list:
#             result.append(item)
    
    return result

def main():
    gd = GoEmotionsDataset(None, "data/emotions.txt")
    gd.load_tsv("data/train.tsv")
    gd.initial_preprocess()
    gd.extract_features()


if __name__ == "__main__":
    main()
