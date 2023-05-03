import string
from pathlib import Path

import gensim.models.fasttext
from gensim.models import KeyedVectors

import click
from gensim.models.fasttext import FastText
import pandas as pd
import spacy
from tqdm import tqdm
from guido.utils.logging_utils import get_custom_logger
from nltk.tokenize import sent_tokenize, word_tokenize

from guido.preprocessor.vectorizers.utils import get_sentences

logger = get_custom_logger("lols")


@click.command(
    help="Given a JSON file with instruction, train fasttext model with gensim"
)
@click.option(
    "--data-path",
    type=Path,
    required=True,
    help="Path to data"
)
def vectorize(data_path: Path):
    sentences = get_sentences(data_path)

    model = FastText(vector_size=300)

    logger.info("build vocabulary")
    # build the vocabulary
    model.build_vocab(sentences)

    logger.info("train fasttext")
    # train the model
    model.train(
        sentences, epochs=model.epochs,
        total_examples=model.corpus_count, total_words=model.corpus_total_words,

    )
    gensim.models.fasttext.Word2Vec
    model.wv.save('.vector_cache/vectors.kv')
    print(model.wv.most_similar("Tipp", topn=10))


if __name__ == "__main__":
    vectorize()
