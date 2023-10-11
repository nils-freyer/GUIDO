import logging
import os
import warnings
from pathlib import Path

import mlflow
import pandas as pd
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import DictConfig
from sklearn.metrics import f1_score
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from tqdm import tqdm
from transformers import BertTokenizerFast
from transformers import TFBertForSequenceClassification, TextClassificationPipeline, PreTrainedTokenizerBase

from guido.utils.logging_utils import get_custom_logger
from guido.preprocessor.utils import replace_url

logger = get_custom_logger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run(conf: DictConfig):
    """

    @param conf:
    @return:
    """
    warnings.filterwarnings("ignore")
    mlflow.tensorflow.autolog()
    model, tokenizer = get_model(lr=conf.optimizer.lr)
    train_data = get_ds(data_path=Path('data/recipes/ger/split/train.csv'),
                        tokenizer=tokenizer,
                        batch_size=conf.training.batch_size)
    test_data = get_ds(data_path=Path("data/recipes/ger/split/test.csv"),
                       tokenizer=tokenizer,
                       batch_size=conf.training.batch_size)
    train(tokenizer=tokenizer,
          model=model,
          train_data=train_data,
          test_data=test_data,
          output_path=Path("weights/sentence_bert"),
          n_epochs=conf.training.epochs)


def get_model(lr: float):
    """

    @param lr: learning rate for optimizer
    @return: tf model and bert tokenizer
    """
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-german-cased')
    model = TFBertForSequenceClassification.from_pretrained("bert-base-german-cased",
                                                            num_labels=2)


    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=tf.metrics.SparseCategoricalAccuracy()
                  )

    return model, tokenizer


def get_ds(data_path: Path, tokenizer: PreTrainedTokenizerBase, batch_size: int = 16, data_type: str = "csv") \
        -> tf.data.Dataset:
    """

    @param data_path:
    @param tokenizer:
    @param batch_size:
    @param data_type: csv, jsonl,...
    @return: dataset
    """
    data = get_data(data_path, data_type)

    X_train, y_train = list(data.data), list(data.label)
    train_encodings = tokenizer(X_train, truncation=True, padding=True)
    train_data = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        y_train
    )).batch(batch_size)

    return train_data


def train(tokenizer: PreTrainedTokenizerBase, train_data: DatasetV2, test_data: DatasetV2,
          model: TFBertForSequenceClassification, output_path: Path, n_epochs: int):
    """
    training the model with given train set, validating during training on test set, validating after training on dev
    set. Saving best model.
    @param tokenizer:
    @param train_data:
    @param test_data:
    @param model:
    @param output_path:
    @param n_epochs:
    @return: None
    """
    metric = 'val_sparse_categorical_accuracy'
    run_name = mlflow.active_run().info.run_name
    checkpoint_path = f"weights/checkpoints/{run_name}"
    early_stopping = EarlyStopping(monitor=metric, patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                 monitor=metric,
                                 verbose=0,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='max')
    model.fit(train_data, epochs=n_epochs, validation_data=test_data, callbacks=[checkpoint, early_stopping])
    f1 = evaluate_f1(model=model, tokenizer=tokenizer, dev_path=Path('data/recipes/ger/split/dev.csv'), data_type='csv')
    logger.info(f"final f1-score of {f1}")
    mlflow.log_metric("final f1", f1)
    tokenizer.save_pretrained(f"{output_path}/{run_name}")
    model.save_pretrained(f"{output_path}/{run_name}")


def predict(model: TFBertForSequenceClassification, text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    """

    @param model:
    @param text:
    @param tokenizer:
    @return: prediction for given sentence
    """
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    prediction = to_numerical(pipe(text)[0]['label'])
    return prediction


def evaluate_f1(dev_path: Path,
                model: TFBertForSequenceClassification,
                tokenizer: PreTrainedTokenizerBase,
                data_type: str = 'csv') -> float:
    """

    @param dev_path:
    @param model:
    @param tokenizer:
    @param data_type:
    @return:
    """
    val = get_data(data_path=dev_path, data_type=data_type)
    logger.info("Evaluating Model")
    predictions = list()
    for text in tqdm(val.data.values):
        prd = predict(model, text, tokenizer)
        predictions.append(prd)
    return f1_score(val.label.values, predictions)


def to_numerical(label: str) -> int:
    if label == 'LABEL_0':
        return 0
    else:
        return 1


def get_data(data_path: Path, data_type: str) -> pd.DataFrame:
    if data_type == 'json':
        data = pd.read_json(data_path)
    elif data_type == 'jsonl':
        data = pd.read_json(data_path, lines=True)
    else:
        data = pd.read_csv(data_path)
    data.data = data.data.apply(replace_url)
    return data
