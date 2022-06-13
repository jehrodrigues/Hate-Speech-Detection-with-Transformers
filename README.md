# Hate Speech Detection with Transformers

This repo aims to perform the textual classification of sentences into hateful or non-hateful text. It fine-tunes a pre-trained transformer model on the Dynamically Generated Hate dataset and evaluates on the HateCheck benchmark.

---

### Contents

* [Installation](#installation)
* [Data](#Data)
* [Train](#Train)
* [Evaluation](#Evaluation)
* [Prediction](#Prediction)
* [Experimentation](#Experimentation)

---

## Installation
```console
$ virtualenv venv -p python3
$ source venv/bin/activate
$ pip install -r requirements.txt
$ pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
```

## Data

### Preprocessing text file

Pre-process external files to generate training, development and test sets.

```console
$ python -m src.data.make_dataset <dataset_file>
```
Parameters:
* **dataset_file**: hate speech dataset (.csv) + binary labels (hate, not-hate), e.g. "DynamicallyHateDataset.csv".

The files must be inside:
```console
$./data/raw/
```

Output:
```console
$./data/processed/
```

## Train
Fine-tune pre-trained transformer models on training and development data.

```console
$ python -m src.models.train_model <model_name>
```
Parameters:
* **model_name**: pre-trained transformer model, e.g. "distilbert-base-uncased" or "bert-base-uncased".

Output:
```console
$./model/
```

## Evaluation

Evaluate transformer models on test data.

```console
$ python -m src.models.evaluate_model <model_name>
```
Parameters:
* **model_name**: transformer model fine-tuned on <dataset_file>, e.g. "checkpoint-32924".

The file must be inside:
```console
$./model/
```

## Prediction

Predict the label of a sentence.

```console
$ python -m src.models.predict_model <model_name> <sentence>
```
Parameters:
* **model_name**: transformer model fine-tuned on <dataset_file>, e.g. "checkpoint-32924".

The file must be inside:
```console
$./model/
```

* **sentence**: sentence to be classified as hateful or non-hateful, e.g. "I hate those women".

## Experimentation

Perform evaluation of Hate Speech Detection models on HateCheck data.

```console
$ cd notebooks/
$ jupyter notebook
```