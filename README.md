# GUIDO: Guideline Discovery & Ordering

This is the repository for the paper [TODO after DOI announcement]().

Provides features:

### Getting started:

Set up dependencies:
```
conda env create -f guido-cuda.yaml
conda activate guido-cuda-test
poetry config virtualenvs.create false
poetry install
spacy download de_core_news_sm
spacy download de_core_news_lg
spacy download de_dep_news_trf

pip install --upgrade git+https://github.com/huggingface/transformers.git
```


##  Sentence Classifier: Data + Training setup

### Training
split labeled dataset:
```
python guido/preprocessor/split_data.py --data-path data/recipes/ger/all.jsonl --out-path data/recipes/ger/split \
--include-blogs False
```
train classifier:
```
python guido/train_and_log.py
```

perform grid search on learning_rate:
```
python guido/train_and_log.py --multirun mlflow.experiment_name=grid_search_
```

visualize results:
```
mlflow ui
```

## Run GUIDO:

```
python guido/main.py --model-path=weights/sentence_bert/<model-name>
```

