# -*- coding: utf-8 -*-
"""
Script used to preprocess datasets files.
"""


def convert_labels(labels):
    """Convert labels into integer format."""
    labels = list(set(labels))
    return {label: idx for idx, label in enumerate(labels)}


class TextPreprocessing(object):
    """
    Handles text pre-processing
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def tokenize_text(self, text):
        """Perform tokenization in a text."""
        return self._tokenizer(text["text"], truncation=True, padding=True)
