import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.dataset.preprocess_helper import preprocess_helper


class GoEmotionsDataset:
    def __init__(self,
                 labels_file: Optional[Union[Path, str]],
                 emotions_path: Union[Path, str],
                 limit: Optional[int] = None,
                 train_cfg=None):
        self.labels: List = None
        self.features = None
        self.train_cfg = train_cfg
        self.data: List = []
        self.limit = limit
        self.names_dict: Dict[str, int] = None
        mlflow.log_artifact(emotions_path)
        with open(emotions_path, encoding="UTF-8") as emotions_file:
            self.emotions = [emotion.split(",") for emotion in emotions_file.readlines()][0]

    # def get_labels(self):
    #     if labels_file:
    #         with open(labels_file, encoding="UTF-8") as labels_fil:
    #             self.label_mapping = json.load(labels_fil)
    #     else:
    #         self.label_mapping = finish

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

    # Maybe to be used at some point.
    # def load_data(self, data_path: Union[Path, str, List[Union[Path, str]]]):
    #     """Method to support multiple datafiles to be used."""
    #     if isinstance(self.index, Path or str):
    #         self.load_tsv(self.index)
    #     elif isinstance(self.index, list):
    #         for idx in self.index:
    #             self.load_tsv(idx)

    def initial_preprocess(self):
        """Apply sequentially cleaning tools from nltk to data."""
        self.data["proc"] = self.data.text.copy()
        for n, func in preprocess_helper.items():
            self.data.proc = self.data.proc.apply(lambda x: func(x))

    def extract_features(self, transform: bool = False):
        """Extract features from preprocessed dataset.

        Args:
            transform: use saved vocabulary to transform or not
        """
        # bag of words
        self.data["proc_t"] = self.data.proc.apply(lambda x: " ".join(x))
        self.data["labels"] = self.data.label.apply(lambda x: int(x.split(',')[0]))
        if not transform:
            # this is case for training dataset
            vectorizer = CountVectorizer()
            features = vectorizer.fit_transform(self.data.proc_t).toarray()
            # dump vocabulary to be used later
            with open(Path(self.train_cfg.CHECKPOINTS_DIR / "features.joblib"), "wb") as buf:
                joblib.dump(vectorizer.vocabulary_, buf)
                mlflow.log_artifact(Path(self.train_cfg.CHECKPOINTS_DIR / "features.joblib"))
        else:
            with open(Path(self.train_cfg.CHECKPOINTS_DIR / "features.joblib"), "rb") as buf:
                vectorizer = CountVectorizer(vocabulary=pickle.load(buf))
                features = vectorizer.transform(self.data.proc_t).toarray()

        dataset = {"features": features, "labels": np.asarray(self.data.labels)}

        return dataset


def main():
    gd = GoEmotionsDataset(None, "data/emotions.txt")
    gd.load_tsv("data/train.tsv")
    gd.initial_preprocess()
    gd.extract_features()


if __name__ == "__main__":
    main()
