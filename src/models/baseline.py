import logging
import argparse
from pathlib import Path
from src.data.read_dataset import get_data
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.data.text_preprocessing import convert_labels

project_dir = Path(__file__).resolve().parents[2]


class BaselinePredict:
    """
    Provides a classic baseline for comparison
    Usage:
    python -m src.models.baseline <sentence>
    """

    def __init__(self, model_name):
        self._model = self.train(model_name)

    def predict(self, sentence: str):
        """Predict the binary class of a sentence using a Logistic Regression
        Args:
            sentence (str): sentence
        Returns:
            binary class (str): hate (class 0) | not-hate (class 1)
        """
        # predict
        test_proba = self._model.predict_proba([sentence])
        test_pred = test_proba.argmax(1).item()

        return test_proba[0], test_pred

    def train(self, model) -> str:
        """Train a logistic regression method"""
        try:
            # pipeline
            pipeline = Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('lr', LogisticRegression(solver="sag"))
            ])

            # data
            df_train, _, _ = get_data()

            # text
            train_texts = df_train['text']

            # labels
            label2int = convert_labels(df_train["label"])
            train_labels = df_train['label'].apply(lambda x: label2int[x])

            # fit
            pipeline.fit(train_texts, train_labels)

            return pipeline

        except Exception:
            logging.error(f'directory or model is invalid or does not exist: {self._model_name}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Parser descriptors
    parser = argparse.ArgumentParser(
        description='''Script used to predict a binary class.''')

    parser.add_argument('sentence',
                        type=str,
                        help='Sentence to be classified as hateful or non-hateful, e.g. "I hate women"')

    args = parser.parse_args()

    logging.info(args.sentence + " - " + str(BaselinePredict("LogisticRegression").predict(args.sentence)))
