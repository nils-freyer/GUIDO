from pathlib import Path

import pandas as pd
from transformers import TFBertForSequenceClassification, BertTokenizerFast

from guido.models.process_extraction.base_miner import BaseMiner, to_table
from guido.models.text_classifier.sentence_bert import predict


class DependencyMiner(BaseMiner):
    def __init__(self, text: str, model_path: Path, file: str, handle_subordinates: bool):
        super().__init__(text=text, handle_subordinates=handle_subordinates)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path, local_files_only=True)
        self.model = TFBertForSequenceClassification.from_pretrained(model_path, local_files_only=True)

        self.file_name = file

    def get_constraints(self, doc):
        tmp_activities = self.activities.copy()
        for idx, act in tmp_activities.items():
            self.clean(idx=idx, act=act, doc=doc)
            self.get_conjuncts(act=act)

    def routine(self, device='cuda', hybrid=True, use_model=True):
        df = pd.DataFrame()
        doc = self.nlp(self.text)
        df['sent_id'] = [sent.start for sent in doc.sents]
        df['sent'] = [sent.text for sent in doc.sents]
        df.to_csv(f"output/{self.file_name}_i2sent.csv")
        total_activities = dict()
        total_relations = dict()
        for sent in doc.sents:
            prd = predict(model=self.model, tokenizer=self.tokenizer, text=sent.text, )
            if prd == 1:
                self.get_matches(doc=sent)
                self.get_constraints(sent)

                for key, act in self.activities.items():
                    total_activities[sent.start + key] = act

                for key, rel_type in self.relations.items():
                    total_relations[(sent.start + key[0], sent.start + key[1])] = rel_type

                tmp_activities = dict()
                for key, act in self.activities.items():
                    tmp_activities[sent.start + key] = act
                # TODO: fix ids for local activities.

                self.constraints[sent.start] = {"text": sent.text,
                                                "activities": tmp_activities.copy().keys()}

                self.relations = dict()
                self.activities = dict()
                self.chunks = dict()
                self.conjuncts = set()
        #        self.get_corefs(doc=coref_doc, doc_constraints=tmp_constraints)
        constraints_jsonl, activities_jsonl, relations_jsonl = to_table(
            total_constraints=self.constraints,
            total_activities=total_activities,
            total_relations=total_relations
        )
        return constraints_jsonl, activities_jsonl, relations_jsonl
