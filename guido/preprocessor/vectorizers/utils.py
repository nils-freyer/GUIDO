import logging
import string
from pathlib import Path
from typing import List

import pandas as pd
from nltk import sent_tokenize, word_tokenize
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_sentences(data_path: Path) -> List[List[str]]:
    data = pd.read_json(data_path)
    sentences = list()
    logger.info("Tokenize sentences")
    uniques = data.Instructions.drop_duplicates(inplace=False).values
    for text in tqdm(uniques):
        sents = sent_tokenize(text)
        tokens = [[token for token in word_tokenize(sent) if token not in string.punctuation] for sent in sents]
        sentences.extend(tokens)
    return sentences
