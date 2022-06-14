import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.data.read_dataset import get_data
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

project_dir = Path(__file__).resolve().parents[2]


class BaselineTraining:
    """
    Provides a classic baseline for comparison
    Usage:
    python -m src.models.baseline
    """

    def __init__(self, algorithm="logistic_regression"):
        self._algorithm_name = algorithm

    def train(self) -> str:
        """Train a logistic regression method"""
        try:
            # pipeline
            pipeline = Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('lr', LogisticRegression(solver="sag"))  # lbfgs
            ])

            # data
            df_train, df_dev, df_test = get_data()

            # text
            train_texts = df_train['text']
            dev_texts = df_test['text']
            test_text = df_test['text']

            # labels
            labels = list(set(df_train['label']))
            label2int = {label: idx for idx, label in enumerate(labels)}
            train_labels = df_train['label'].apply(lambda x: label2int[x])
            dev_labels = df_dev['label'].apply(lambda x: label2int[x])
            test_labels = df_test['label'].apply(lambda x: label2int[x])

            # fit
            pipeline.fit(train_texts, train_labels)

            # predict
            test_pred = pipeline.predict(test_text)

            # evaluate
            baseline_accuracy = np.mean(test_pred == test_labels)
            print("Baseline accuracy:", baseline_accuracy)

            cm = confusion_matrix(test_labels, test_pred)

            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.show()
        except Exception:
            logging.error(f'directory or model is invalid or does not exist: {self._algorithm_name}')

if __name__ == '__main__':
    BaselineTraining().train()
