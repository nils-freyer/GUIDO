from enum import Enum
from typing import List, Tuple
import pandas as pd


class Relation(Enum):
    AND = 1
    OR = 2


class SemanticComponent:
    def __init__(self, text: str, idx: int):
        self.text = text
        self.idx = idx

    def __hash__(self):
        return self.idx


class Modal:
    def __init__(self, modifier: SemanticComponent, noun: SemanticComponent = None):
        self.noun = noun
        self.modifier = modifier


class Activity:
    def __init__(self, verbs: List[int], doc=None):
        self.doc = doc
        self.verbs = verbs
        self.mods = set()
        self.subjects = set()
        self.objects = set()
        self.subjects_chunks = set()
        self.objects_chunks = set()
        self.aux = set()
        self.pds = set()
        self.cvc = set()
        self.is_neg = False
        self.parallel = False
        self.optional = False

    def set_neg(self):
        self.is_neg = True

    def set_optional(self):
        self.optional = True

    def set_parallel(self):
        self.parallel = True

    def get_chunk_str(self):
        subject_chunks_str = []
        for chunk in self.subjects_chunks:
            chunk_str = ""
            for idx in chunk:
                chunk_str += str(self.doc[idx].text)
            subject_chunks_str.append(chunk_str)

        object_chunks_str = []
        for chunk in self.objects_chunks:
            chunk_str = ""
            for idx in chunk:
                chunk_str += str(self.doc[idx].text)
            object_chunks_str.append(chunk_str)

        modifier_chunk_str = []
        for mod in self.mods:
            if mod.noun is not None:
                mod_str = f"{mod.modifier.text} {mod.noun.text}"
            else:
                mod_str = f"{mod.modifier.text}"
            modifier_chunk_str.append(mod_str)

        cvc_chunk_str = []
        for cvc in self.cvc:
            cvc_str = f"{cvc.modifier.text} {cvc.noun.text}"
            cvc_chunk_str.append(cvc_str)

        return subject_chunks_str, object_chunks_str, modifier_chunk_str, cvc_chunk_str

    def to_str_dict(self):

        subject_chunks_str, object_chunks_str, mod_chunk_str, cvc_chunk_str = self.get_chunk_str()

        return dict(
            Activity=([self.doc[verb].text for verb in self.verbs],
                      [self.doc[aux].text for aux in self.aux],
                      [self.doc[pd].text for pd in self.pds]),
            CVC=cvc_chunk_str,
            Negative=self.is_neg,
            Subjects=[self.doc[sb].text for sb in self.subjects] + subject_chunks_str,
            Objects=[self.doc[ob].text for ob in self.objects] + object_chunks_str,
            Modifiers=mod_chunk_str,
            Parallel=self.parallel,
            Optional=self.optional
        )

    def __str__(self):
        subject_chunks_str, object_chunks_str, mod_str, cvc_str = self.get_chunk_str()
        out_str = f"Activity:{[self.doc[verb].text for verb in self.verbs]}, " \
                  f"{[self.doc[aux].text for aux in self.aux]}, " \
                  f"neg: {self.is_neg}, " \
                  f"predicates: {[self.doc[pd].text for pd in self.pds]}"
        out_str += "\n"
        out_str += "Subjects:"
        for subject in self.subjects:
            out_str += f" {self.doc[subject].text}; "
        for chunk in subject_chunks_str:
            out_str += chunk
        out_str += "\n"
        out_str += "Objects:"
        for ob in self.objects:
            out_str += f" {self.doc[ob].text}; "
        for chunk in object_chunks_str:
            out_str += chunk
        out_str += "\n"
        out_str += "Modifiers:"
        for mod in self.mods:
            out_str += f" {mod.modifier.text} {mod.noun.text}; "
        return out_str


