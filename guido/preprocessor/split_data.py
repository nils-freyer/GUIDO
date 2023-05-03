from pathlib import Path

import click
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split

from guido.utils.logging_utils import get_custom_logger

logger = get_custom_logger(__name__)


@click.command(
    help="Given a CSV file, split the file into train, test and dev"
)
@click.option(
    "--data-path",
    type=Path,
    required=True,
    help="Path to data to be split into train, test and dev files"
)
@click.option(
    "--out-path",
    type=Path,
    required=True,
    help="Path to data to be split into train, test and dev files"
)
@click.option(
    "--balance",
    type=bool,
    required=False,
    default=True,
    help="True if data should be balanced"
)
@click.option(
    "--data-type",
    type=str,
    required=False,
    default='jsonl',
    help="Datatype of dataset"
)
@click.option(
    "--include-blogs",
    type=bool,
    required=False,
    default=False,
    help="If true, use the blog data too."
)
@click.option(
    "--blog-path",
    type=Path,
    required=False,
    default=Path("data/recipes/ger/blogs.jsonl"),
    help="Path to blog data"
)
def split(data_path: Path,
          out_path: Path,
          balance: bool,
          data_type: str,
          include_blogs: bool,
          blog_path: Path,
          process_label='process',
          irrelevant_label='irrelevant'):
    """
    read dataset for specified data path, drop non-annotated texts and save train, test and dev set.
    :param data_type:
    :param out_path: dir to save the split data
    :param data_path: path to dataset
    :param balance: if balance==True, balance the dataset
    :param include_blogs: if true, use blog posts
    :param blog_path: path to blog data
    :param process_label: label for texts with process information
    :param irrelevant_label: label for texts with irrelevant information
    :return: void
    """

    # Read the labeled dataset:
    logger.info(f"Read dataset at {data_path} of type {data_type}")
    if data_type == 'jsonl':
        df = pd.read_json(data_path, lines=True)
    elif data_type == 'json':
        df = pd.read_json(data_path)
    else:
        df = pd.read_csv(data_path)
    logger.info(f"Total length of dataset: {len(df)}")
    df = df[df.label.notna()][["data", "label"]]

    # load blog posts if include_blogs==True:
    if include_blogs:
        logger.info("Load blog data")
        blogs = pd.read_json(blog_path, lines=True)
        nlp = spacy.load("de_dep_news_trf", exclude="ner")
        sentences = []
        for text in blogs.text.values:
            doc = nlp(text)
            for sent in doc.sents:
                sentences.append(sent.text)
        logger.info(f"Include {len(sentences)} sentences from blog posts")
        label = ['irrelevant' for _ in range(len(sentences))]
        df_blogs = pd.DataFrame({'data': sentences, 'label': label})
        df_blogs = df_blogs[[len(text) > 3 for text in df_blogs.data.values]]
        df = pd.concat([df, df_blogs])

    # make labels numerical
    df.label = df.label.apply(encode_label)
    df.dropna(inplace=True)
    # remove punctuation sentences and broken sentences by length heuristic:
    df = df[df.label < 2]
    logger.info(f"Length of annotated dataset: {len(df)}")
    logger.info(f"# rows with text len <= 0: {len(df[[len(text) == 3 for text in df.data.values]])}")
    # balance dataset if balance==True
    if balance:
        df = balance_data(data=df)
        logger.info(f"Length of balanced dataset: {len(df)}")

    # TODO: should this be parametrized?
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_dev = train_test_split(df_train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2
    logger.info(f"Length of train set: {len(df_train)}")
    logger.info(f"Length of test set: {len(df_test)}")
    logger.info(f"Length of dev set: {len(df_dev)}")
    # save datasets to specified output path
    logger.info(f"Save split datasets to {out_path}")
    df_train.to_csv(out_path / "train.csv")
    df_test.to_csv(out_path / "test.csv")
    df_dev.to_csv(out_path / "dev.csv")


def balance_data(data: pd.DataFrame) -> pd.DataFrame:
    df_other = data[data.label == 0]
    df_process = data[data.label == 1]
    balance_count = min(len(df_process), len(df_other))
    df_other = df_other.sample(balance_count)
    df_process = df_process.sample(balance_count)
    df_balanced = pd.concat([df_process, df_other])
    return df_balanced


def encode_label(label):
    if 'irrelevant' in label:
        return 0
    elif 'process' in label:
        return 1
    else:
        return 2


if __name__ == "__main__":
    split()
