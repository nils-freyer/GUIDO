from pathlib import Path

import click

from guido.utils.bulletpoints import make_bulletpoints
from guido.utils.logging_utils import get_custom_logger
from guido.models.process_extraction.dependency_miner.dep_miner import DependencyMiner
from guido.preprocessor.utils import replace_temporals
from guido.process_model.petri_net import make_net
from guido.utils.utils import write_jsons, get_data

logger = get_custom_logger(__name__)


@click.command(
    help="Given a recipe, create a petri net."
)
@click.option(
    "--data-path",
    type=Path,
    required=False,
    default="data/recipes/ger/split/recipes_test.json",
    help="Path to data to be split into train, test and dev files"
)
@click.option(
    "--data_type",
    type=str,
    default='json',
    required=False,
    help="Data type"
)
@click.option(
    "--text-column",
    type=str,
    required=True,
    default="Instructions",
    help="Dataframe column to read the recipe text"
)
@click.option(
    "--model-path",
    type=Path,
    required=True,
    help="Path to huggingface model"
)
@click.option(
    "--handle-subordinates",
    type=bool,
    required=True,
    default=True,
    help="If true, use vvimp heuristic to handle subordinates"
)
def cooking(data_path: Path, data_type: str, model_path: Path, text_column: str, handle_subordinates: bool):
    logger.info(f"Reading {data_path} of type {data_type}")
    df = get_data(data_path, data_type)
    N = len(df)
    logger.info(f"Converting {N} texts from {data_path} to WF Nets")
    for i in range(N):
        logger.info(f"Converting the {i}th text")
        text = replace_temporals(text=df[text_column].values[i])

        file = f'recipe_{i}'

        logger.info(f"Initializing Miner with model from {model_path}")
        miner = DependencyMiner(
            text=text,
            model_path=model_path,
            file=file,
            handle_subordinates=handle_subordinates
        )
        logger.info("Extracting process steps and order")
        constraints_jsonl, activities_jsonl, relations_jsonl = miner.routine()
        if len(constraints_jsonl) == 0:
            continue
        logger.info("Saving constraints, activities, and relations")
        write_jsons(activities_jsonl, constraints_jsonl, file, relations_jsonl)

        logger.info("Making petri net")

        make_net(file=file)

        logger.info("Saving markdown")
        make_bulletpoints(file=file, petri_path=f'../petri_nets/{file}_pn.png')





if __name__ == "__main__":
    cooking()

"""
ROOT  --  root
ac  --  adpositional case marker
adc  --  adjective component
ag  --  genitive attribute
ams  --  measure argument of adjective
app  --  apposition
avc  --  adverbial phrase component
cc  --  coordinating conjunction
cd  --  coordinating conjunction
cj  --  conjunct
cm  --  comparative conjunction
cp  --  complementizer
cvc  --  collocational verb construction
da  --  dative
dep  --  unclassified dependent
dm  --  discourse marker
ep  --  expletive es
ju  --  junctor
mnr  --  postnominal modifier
mo  --  modifier
ng  --  negation
nk  --  noun kernel element
nmc  --  numerical component
oa  --  accusative object
oc  --  clausal object
og  --  genitive object
op  --  prepositional object
par  --  parenthetical element
pd  --  predicate
pg  --  phrasal genitive
ph  --  placeholder
pm  --  morphological particle
pnc  --  proper noun component
punct  --  punctuation
rc  --  relative clause
re  --  repeated element
rs  --  reported speech
sb  --  subject
sbp  --  passivized subject (PP)
svp  --  separable verb prefix
uc  --  unit component
vo  --  vocative
"""
