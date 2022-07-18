import argparse
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from src.dataset.preprocess_helper import preprocess_helper


def run_inference(text, model_type: str):
    logger = logging.getLogger(__name__)
    # TODO connect somehow the models with models registry of mlflow (maybe)?

    if model_type == "RandomForest":
        model_path = Path("workdir/checkpoints/RF/random_forest_model_50.joblib")
        features_dump = Path("workdir/checkpoints/RF/features.joblib")
    elif model_type == "LogisticRegression":
        model_path = Path("workdir/checkpoints/LR/logistic_regression_model_50.joblib")
        features_dump = Path("workdir/checkpoints/LR/features.joblib")

    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}.")

    with open(Path("data/emotions.txt"), encoding="UTF-8") as emotions_file:
        emotions = [emotion.split("\n")[0] for emotion in emotions_file.readlines()]
        logger.info("Emotions loaded from data/emotions.txt")

    with open(features_dump, "rb") as buf:
        vectorizer = CountVectorizer(vocabulary=joblib.load(buf))
        features = vectorizer.transform([text]).toarray()
        voc = vectorizer.vocabulary
        logger.info(f"Vocabulary loaded from {features_dump}.")

    # preprocess
    # for _, func in preprocess_helper.items():
    #     text = func(text)
    # text = " ".join(text)

    # get predictions and confidences
    predictions = model.predict_proba(features).flatten()
    confidences = {emotions[i]: float(predictions[i]) for i in range(len(emotions))}

    def preprocess_data(textt):
        textt = textt.lower()
        for _, func in preprocess_helper.items():
            textt = func(textt)
        with open(features_dump, "rb") as buf:
            vectorizer = CountVectorizer(vocabulary=joblib.load(buf))
            features = vectorizer.transform(textt).toarray()
        return features

    # train_cfg = get_training_config()
    # dataset = GoEmotionsDataset(train_cfg.LABELS_FILE, train_cfg.EMOTIONS_FILE, limit=20, train_cfg=train_cfg)
    # dataset.load_tsv("data/train.tsv")
    # dataset.initial_preprocess()
    # train_dataloader = dataset.extract_features(transform=True)

    if model_type == "RandomForest":
        weights = model.feature_importances_
    else:
        weights = np.sum(model.coef_, axis=0)
    feats = dict(zip(list(dict(sorted(voc.items(), key=lambda item: item[1])).keys()), weights))
    xs = np.array(list(dict(sorted(voc.items(), key=lambda item: item[1]))))[np.where(preprocess_data(text))[1]]
    ys = []
    for xsi in xs:
        ys.append(feats[xsi])

    # produce output figures
    _, ax = plt.subplots(2, 1)
    ax[0].set_title("Strength of known words.")
    ax[0].bar(xs, ys)
    ax[0].tick_params(axis="x", rotation=45)

    ax[1].set_title("Insight into how representative was the training set of words.")
    ax[1].bar(["Known", "Unknown"], [sum(np.sum(preprocess_data(text), axis=0)),
                                     len(text.split(" ")) - sum(np.sum(preprocess_data(text), axis=0))])
    ax[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    figure = plt.gcf()

    return confidences, figure


def main():
    parser = argparse.ArgumentParser(description="Emotion classifier inference.")
    parser.add_argument("--verbose_level", "-v", choices=["debug", "info", "error"], default="info", type=str,
                        help="Logger level.")
    parser.add_argument("--text", type=str, help="Text to run inference over.")
    parser.add_argument("--model", type=str, help="Model name to use for inference (same as in configuration).")
    args = parser.parse_args()

    verbose_level: str = args.verbose_level
    text: str = args.text
    model: str = args.model
    if text is None:
        text = "Reddit is a network of communities where people can dive into their interests, hobbies and passions."
    if model is None:
        model = "RandomForest"

    conf, _ = run_inference(text, model)

    print(conf)
    plt.show()


if __name__ == "__main__":
    main()
