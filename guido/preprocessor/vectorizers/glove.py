import csv
import pickle
from pathlib import Path

import click
import nltk.corpus
import numpy as np
from collections import Counter
import logging
import pandas as pd
import spacy
from nltk import sent_tokenize
from nltk.corpus import brown
from mittens import GloVe, Mittens
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from functools import reduce
from gensim.scripts.glove2word2vec import glove2word2vec
from guido.utils.logging_utils import get_custom_logger
from guido.preprocessor.vectorizers.utils import get_sentences

logger = get_custom_logger(__name__)


def glove2dict(glove_filename):
    with open(glove_filename, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:])))
                 for line in reader}
    return embed


#
def get_rare_oov(xdict, val):
    return [k for (k, v) in Counter(xdict).items() if v <= val]


@click.command(
    help="Given a CSV file, split the file into train, test and dev"
)
@click.option(
    "--glove-path",
    type=Path,
    required=True,
    help="Path to pre-trained glove embeddings"
)
@click.option(
    "--data-path",
    type=Path,
    required=True,
    help="Path to fine tune pre-trained embeddings on"
)
@click.option(
    "--output-path",
    type=Path,
    required=True,
    help="output path"
)
@click.option(
    "--dim",
    type=int,
    required=False,
    default=300,
    help="dimensionality of the pre-trained glove embedding"
)
@click.option(
    "--remove-sw",
    type=bool,
    required=False,
    default=True,
    help="If true, stopwords will be removed"
)
@click.option(
    "--n-iter",
    type=int,
    required=False,
    default=1000,
    help="Maximum training iterations"
)
def fine_tune_glove(glove_path: Path, data_path: Path, output_path: Path, dim: int, remove_sw: bool, n_iter: int):
    """

    @param n_iter:
    @param remove_sw: if true, remove stopwords (german)
    @param dim: dimensionality of pre-trained glove embeddings
    @param data_path: path to dataset to finetune on
    @param output_path: path to save output embeddings to
    @param glove_path: path to pre-trained glove embeddings
    """
    logger.info(f"Loading embeddings from {glove_path}")
    pre_glove = glove2dict(glove_filename=glove_path)
    logger.info(f"Size of initial vocab: {len(pre_glove.keys())}")
    sw = list()
    if remove_sw:
        sw = nltk.corpus.stopwords.words('german')
    logger.info(f"Reading and preprocessing dataset from {data_path}")
    sentences = get_sentences(data_path=data_path)

    tokens = reduce(lambda a, b: a + b, sentences)
    recipe_nonstop = [token.lower() for token in tqdm(tokens) if (token.lower() not in sw)]
    vocab_rare = get_rare_oov(recipe_nonstop, 1)
    tokens = [token for token in tqdm(recipe_nonstop) if token not in vocab_rare]
#    unseen_embeddings = [(k, v) for k, v in pre_glove.items() if k not in tokens]
    vocab = set(tokens)
    recipe_doc = [' '.join(tokens)]

    logger.info("Creating co-occurrence matrix")
    cv = CountVectorizer(ngram_range=(1, 1), vocabulary=vocab)
    X = cv.fit_transform(recipe_doc)
    Xc = (X.T * X)
    Xc.setdiag(0)
    co_occ_ar = Xc.toarray()

    mittens_model = Mittens(n=dim, max_iter=n_iter)

    logger.info("Fine tune Glove model")
    fine_tuned_embeddings = mittens_model.fit(
        co_occ_ar,
        vocab=vocab,
        initial_embedding_dict=pre_glove)

    print('\n')

    logger.info(f"Size of final vocab: {len(vocab)}")
    logger.info(f"Save fine tuned embeddings to {output_path}")
    fine_tuned_glove = dict(zip(list(vocab), fine_tuned_embeddings))
    with open(output_path, "w") as f:
        for k, v in fine_tuned_glove.items():
            f.write(k)
            for u in v:
                f.write(" ")
                f.write(f"{u}")
            f.write('\n')

    glove2word2vec(glove_input_file=str(output_path), word2vec_output_file=str(output_path).replace("pt", "txt"))


if __name__ == "__main__":
    fine_tune_glove()
