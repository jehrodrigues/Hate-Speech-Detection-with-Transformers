# -*- coding: utf-8 -*-
"""
Script used to read datasets files.
"""
import pandas as pd
import logging
from typing import Tuple
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]


def split_data(data, labels, train_frac, random_state=None) -> Tuple[str, str]:
    """
    param data: Data to be split
    param labels: labels to be used on stratify
    param train_frac: Ratio of train set to whole dataset

    Randomly split dataset, based on these ratios:
        'train': train_frac
        'test': 1-train_frac

    Eg: passing train_frac=0.8 gives a 80% / 20% split
    """
    pass


def balance_data(df_dataset) -> pd.DataFrame:
    """
    param data: Dataframe to be balance

    Downsample majority class equal to the number of samples in the minority class
    """
    pass


def get_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reads datasets."""
    path = project_dir / 'data' / 'processed'
    if path.exists():
        try:
            df_train = pd.read_csv(path / 'train.csv', delimiter=",",
                                   header=0, encoding='utf-8', engine='python')

            df_dev = pd.read_csv(path / 'dev.csv', delimiter=",",
                                 header=0, encoding='utf-8', engine='python')

            df_test = pd.read_csv(path / 'test.csv', delimiter=",",
                                  header=0, encoding='utf-8', engine='python')
        except pd.errors.EmptyDataError:
            logging.error(f'file is empty and has been skipped.')
        return df_train, df_dev, df_test


class DatasetReader:
    """Handles dataset reading"""

    def __init__(self):
        pass
