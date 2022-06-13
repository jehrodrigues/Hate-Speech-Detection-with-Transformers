# -*- coding: utf-8 -*-
"""
Script used to read external files in order to generate training, development and test sets.
Usage:
python -m src.data.make_dataset <dataset_file> <testset_file>
"""
import argparse
import logging
import pandas as pd
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]


def create_train_test_sets(dataset_file: str) -> None:
    """
    Create training and test sets with the same distribution for language pair and accuracy (class), in order to train and evaluate trained models

    param dataset_file: Data to be split
    """
    path = project_dir / 'data' / 'processed'
    df_dataset = pd.read_csv(dataset_file, delimiter=",",
                             usecols=['text', 'label', 'split'],
                             header=0, encoding='utf-8', engine='python')

    # split into train, dev and test
    df_train = df_dataset[df_dataset['split'] == 'train']
    df_dev = df_dataset[df_dataset['split'] == 'dev']
    df_test = df_dataset[df_dataset['split'] == 'test']

    # Resume
    logging.info('\ntrain-------------------------------------------------------------')
    logging.info(df_train.shape)
    logging.info('label     %')
    logging.info(f" {round(df_train.groupby('label')['text'].count() * 100 / df_train.shape[0], 2)}")

    logging.info('\ndev-------------------------------------------------------------')
    logging.info(df_dev.shape)
    logging.info('label     %')
    logging.info(f" {round(df_dev.groupby('label')['text'].count() * 100 / df_dev.shape[0], 2)}")

    logging.info('\ntest-------------------------------------------------------------')
    logging.info(df_test.shape)
    logging.info('label     %')
    logging.info(f" {round(df_test.groupby('label')['text'].count() * 100 / df_test.shape[0], 2)}")

    # Save files
    df_train.to_csv(path / 'train.csv', index=False)
    df_dev.to_csv(path / 'dev.csv', index=False)
    df_test.to_csv(path / 'test.csv', index=False)


if __name__ == '__main__':
    # Parser descriptors
    parser = argparse.ArgumentParser(
        description='''Script used to read external files in order to generate training, development and test sets.''')

    parser.add_argument('dataset_file',
                        help='Choose the dataset to be split into train, dev and test. The file must be inside the ./data/raw/ directory and the extension must be .csv: {dataset.csv}.')

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    args = parser.parse_args()

    dataset_path = project_dir / 'data' / 'raw' / args.dataset_file
    create_train_test_sets(dataset_path)
