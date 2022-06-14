# -*- coding: utf-8 -*-
"""
Script used to predict a class through transformers models.
Usage:
    python -m src.models.predict_model <model_name> <sentence>
"""
import logging
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

project_dir = Path(__file__).resolve().parents[2]


class TransformerPredict:
    """Provides a prediction based on transformers and pre-trained models"""

    def __init__(self, model_name):
        self._model_name = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict(self, sentence: str) -> str:
        """Predict the binary class of a sentence using a Transformer model
        Args:
            sentence (str): sentence
        Returns:
            binary class (str): hate (class 0) | not-hate (class 1)
        """
        try:
            # Define the pipeline
            pipeline = TextClassificationPipeline(model=self._model_name, tokenizer=self._tokenizer, return_all_scores=True)
            return pipeline(sentence)

        except Exception:
            logging.error(f'directory or model is invalid or does not exist: {self._model_name}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Parser descriptors
    parser = argparse.ArgumentParser(
        description='''Script used to predict a binary class.''')

    parser.add_argument('model_name',
                        type=str,
                        default='checkpoint-32924',
                        help='Fine-tuned transformer model, e.g. checkpoint-32924')

    parser.add_argument('sentence',
                        type=str,
                        help='Sentence to be classified as hateful or non-hateful, e.g. "I hate women"')

    args = parser.parse_args()

    model_path = project_dir / 'model' / args.model_name

    logging.info(args.sentence + " - " + str(TransformerPredict(model_path).predict(args.sentence)))
