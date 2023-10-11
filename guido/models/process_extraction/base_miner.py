from abc import abstractmethod
from copy import copy

import spacy
from spacy.matcher import DependencyMatcher

from guido.models.process_extraction.matcher_utils import V_PATTERN_OA, V_PATTERN_OP, V_PATTERN_OC, V_PATTERN_CVC, \
    V_PATTERN_OG, V_PATTERN_SBP, V_PATTERN_SB, A_PATTERN_OA, A_PATTERN_OP, A_PATTERN_OC, \
    A_PATTERN_OG, A_PATTERN_SBP, A_PATTERN_SB, A_PATTERN_NEG, A_PATTERN_PD, V_PATTERN_MO, \
    N_PATTERN_AG, N_PATTERN_PG, N_PATTERN_CJ, A_PATTERN_MO, V_PATTERN_CJ, V_PATTERN_NEG, \
    N_PATTERN_CJ_CD, PN_PATTERN_CJ, A_PATTERN_CJ, V_PATTERN_CJ_CD, CONJ, TMP_BEFORE, TMP_AFTER, \
    TMPS_PARALLEL, MARKERS_EXISTS
from guido.models.process_extraction.semantic_components import SemanticComponent, Activity, Modal

USE_CONJ = False


def merge_chunks(act: Activity, chunks: list):
    temp_subjects = act.subjects.copy()
    temp_objects = act.objects.copy()

    # check activity subjects for elements included in chunks
    for sb in temp_subjects:
        fits = [set(chunk) for chunk in chunks if sb in chunk]
        if len(fits) > 0:
            act.subjects.remove(sb)
            for chunk in fits:
                act.subjects_chunks.update(zip(chunk))

    # check activity objects for elements included in chunks
    for ob in temp_objects:
        fits = [set(chunk) for chunk in chunks if ob in chunk]
        if len(fits) > 0:
            act.objects.remove(ob)
            for chunk in fits:
                act.objects_chunks.update(zip(chunk))


def to_table(total_constraints: dict, total_activities: dict, total_relations: dict):
    constraints_jsonl = list()
    activities_jsonl = list()
    relations_jsonl = list()

    for sent_id, constraints in total_constraints.items():
        for act_id in constraints["activities"]:
            line = dict()
            line["sent_id"] = sent_id
            line["act_id"] = act_id
            line["text"] = constraints["text"]
            constraints_jsonl.append(line)

    for act_id, act in total_activities.items():
        line = act.to_str_dict()
        line["id"] = act_id
        activities_jsonl.append(line)

    for key, rel in total_relations.items():
        line = dict()
        line['lact'] = key[0]
        line['ract'] = key[1]
        line['type'] = rel
        relations_jsonl.append(line)

    return constraints_jsonl, activities_jsonl, relations_jsonl


class BaseMiner:
    def __init__(self, text: str, handle_subordinates: bool = True):
        # inits for dependency parsing
        self.handle_subordinates = handle_subordinates
        self.nlp = spacy.load('de_dep_news_trf')

        self.nlp.add_pipe("merge_noun_chunks")
        self.matcher = DependencyMatcher(self.nlp.vocab)
        self.matcher.add("CHUNKS_PG", [N_PATTERN_PG])
        self.matcher.add("CHUNKS_AG", [N_PATTERN_AG], )
        self.matcher.add("OA", [V_PATTERN_OA, A_PATTERN_OA], on_match=self.add_oa)
        self.matcher.add("OP", [V_PATTERN_OP, A_PATTERN_OP], on_match=self.add_op)
        self.matcher.add("OG", [V_PATTERN_OG, A_PATTERN_OG], on_match=self.add_og)
        self.matcher.add("SB", [V_PATTERN_SB, A_PATTERN_SB], on_match=self.add_sub)
        # TODO: passivized subjects for noun based activities
        self.matcher.add("SBP", [V_PATTERN_SBP, A_PATTERN_SBP])
        self.matcher.add("NEG", [A_PATTERN_NEG, V_PATTERN_NEG], on_match=self.add_neg)
        self.matcher.add("MO", [V_PATTERN_MO, A_PATTERN_MO], on_match=self.add_mod)
        self.matcher.add("PD", [A_PATTERN_PD], on_match=self.add_pd)
        self.matcher.add("AUX", [A_PATTERN_OC], on_match=self.add_aux)
        self.matcher.add("CVC", [V_PATTERN_CVC], on_match=self.add_cvc)

        self.matcher.add("NCJ", [N_PATTERN_CJ, PN_PATTERN_CJ], on_match=self.add_ncj)
        self.matcher.add("NCDJ", [N_PATTERN_CJ_CD], on_match=self.add_ncdj)
        self.matcher.add("VCJ", [V_PATTERN_CJ, A_PATTERN_CJ], on_match=self.add_vcj)
        self.matcher.add("VCJ", [V_PATTERN_CJ_CD], on_match=self.add_vcj)
        self.matcher.add("VERB", [V_PATTERN_OC], on_match=self.add_verb)

        # inits for returns
        self.relations = dict()
        self.activities = dict()
        self.chunks = dict()
        self.conjuncts = set()
        self.v_conjuncts = set()
        self.v_disjuncts = set()
        self.conjuncts_rel = dict()
        self.constraints = dict()
        self.reversed = set()
        self.parallel = set()
        self.text = text


    def add_ncj(self, matcher, doc, i, matches):
        match_id, [cj1, cj2] = matches[i]
        self.conjuncts.add((cj1, cj2))

    def add_ncdj(self, matcher, doc, i, matches):
        match_id, [cj1, cd, cj2] = matches[i]
        self.conjuncts.add((cj1, cj2))

    def add_vcj(self, matcher, doc, i, matches):
        if len(matches[i][1]) == 2:
            match_id, [cj1, cj2] = matches[i]
            pre = min([cj1, cj2])
            post = max([cj1, cj2])
            if set([child.lemma_ for child in doc[post].children]) & set(TMP_BEFORE) or set(
                    [child.lemma_ for child in doc[pre].children]) & set(TMP_AFTER):
                self.relations[(cj1, cj2)] = "rev"
            elif set([child.lemma_ for child in doc[post].children]) & set(TMPS_PARALLEL):
                self.relations[(cj1, cj2)] = "conj"
            self.make_conjunct(cj1, cj2, doc)

        else:
            match_id, [cj1, cd, cj2] = matches[i]
            cd_text = doc[cd].text
            if cd_text in CONJ:
                self.check_and_set_conjunction(cj1, cj2, doc)
            else:
                self.relations[(cj1, cj2)] = "disj"
            self.make_conjunct(cj1, cj2, doc)

    def add_op(self, matcher, doc, i, matches):
        pass

    def add_pd(self, matcher, doc, i, matches):
        match_id, [verb_i, pd_i] = matches[i]

        if verb_i not in self.activities.keys():
            act = Activity(verbs=[verb_i], doc=doc)
            act.pds.add(pd_i)
            self.activities[verb_i] = act
        else:
            self.activities[verb_i].pds.add(pd_i)

    def add_og(self, matcher, doc, i, matches):
        pass

    def add_sub(self, matcher, doc, i, matches):
        match_id, [verb_i, sb_i] = matches[i]

        if verb_i not in self.activities.keys():
            act = Activity(verbs=[verb_i], doc=doc)
            act.subjects.add(sb_i)
            self.activities[verb_i] = act
        else:
            self.activities[verb_i].subjects.add(sb_i)

    def add_oa(self, matcher, doc, i, matches):
        match_id, [verb_i, oa_i] = matches[i]

        if verb_i not in self.activities.keys():
            act = Activity(verbs=[verb_i], doc=doc)
            act.objects.add(oa_i)
            self.activities[verb_i] = act
        else:
            self.activities[verb_i].objects.add(oa_i)


    def add_aux(self, matcher, doc, i, matches):
        match_id, [verb1_i, verb2_i] = matches[i]

        if doc[verb1_i].pos_ == "AUX":
            aux_i = verb1_i
            verb_i = verb2_i
        else:
            verb_i = verb1_i
            aux_i = verb2_i

        if verb_i not in self.activities.keys():
            act = Activity(verbs=[verb_i], doc=doc)
            act.aux.add(aux_i)
            self.activities[verb_i] = act
        else:
            self.activities[verb_i].aux.add(aux_i)

    def add_verb(self, matcher, doc, i, matches):
        match_id, [verb1_i, verb2_i] = matches[i]
        if verb1_i in self.activities.keys():
            self.activities[verb1_i].verbs.append(verb2_i)
        elif verb2_i in self.activities.keys():
            self.activities[verb2_i].verbs.append(verb1_i)

    def add_neg(self, matcher, doc, i, matches):
        match_id, [verb_i, neg_i] = matches[i]
        if verb_i not in self.activities.keys():
            act = Activity(verbs=[verb_i], doc=doc)
            self.activities[verb_i] = act

        self.activities[verb_i].set_neg()

    def add_cvc(self, matcher, doc, i, matches):
        match_id, [verb_i, cvc, nk] = matches[i]
        mod = SemanticComponent(text=doc[cvc], idx=cvc)
        noun = SemanticComponent(text=doc[nk], idx=nk)
        if verb_i not in self.activities.keys():
            act = Activity(verbs=[verb_i], doc=doc)
            self.activities[verb_i] = act

        self.activities[verb_i].cvc.add(Modal(modifier=mod, noun=noun))

    def add_mod(self, matcher, doc, i, matches):
        # TODO: add option for conjuncts
        match_id, [verb_i, mo_i] = matches[i]
        mod = SemanticComponent(text=doc[mo_i], idx=mo_i)
        nk_children = [nk for nk in doc[mo_i].children if nk.dep_ == 'nk']
        if len(nk_children) > 0:
            nk_i = nk_children[0].i
            if verb_i not in self.activities.keys():
                act = Activity(verbs=[verb_i], doc=doc)
                noun = SemanticComponent(text=doc[nk_i], idx=nk_i)
                self.activities[verb_i] = act
                self.activities[verb_i].mods.add(Modal(modifier=mod, noun=noun))
        else:
            if verb_i not in self.activities.keys():
                act = Activity(verbs=[verb_i], doc=doc)
                self.activities[verb_i] = act
                self.activities[verb_i].mods.add(Modal(modifier=mod))

    def make_conjunct(self, cj1: int, cj2: int, doc):
        if cj1 in self.activities.keys() and cj2 not in self.activities.keys():
            act = copy(self.activities[cj1])
            act.verbs = [cj2]
            self.activities[cj2] = act

        elif cj2 in self.activities.keys() and cj1 not in self.activities.keys():
            act = copy(self.activities[cj2])
            act.verbs = [cj1]
            self.activities[cj1] = act
        elif cj1 not in self.activities.keys() and cj2 not in self.activities.keys():
            act_1 = Activity(verbs=[cj1], doc=doc)
            self.activities[cj1] = act_1

            act_2 = Activity(verbs=[cj2], doc=doc)
            self.activities[cj2] = act_2

    def clean(self, idx: int, act: Activity, doc):
        children = [child.lemma_.lower() for child in doc[idx].children]
        self.check_and_set_parallel(children, idx)
        self.check_and_set_optional(children, idx)
        if doc[idx].pos_ == "AUX":
            if doc[idx].text in MARKERS_EXISTS:
                act.set_optional()
            self.merge_aux(act, idx)
        else:
            self.merge_overlaps(act, idx)
        if self.handle_subordinates:
            if doc[idx].tag_ != 'VVIMP':
                for i in self.activities.keys():
                    if doc[i].tag_ == 'VVIMP' and idx in self.activities.keys():
                        self.activities.pop(idx)
                        break

    def merge_overlaps(self, act, idx):
        potential_overlaps = [alt_act for i, alt_act in self.activities.items() if
                              set(act.verbs).issubset(set(alt_act.verbs)) and i != idx]
        for overlap in potential_overlaps:
            overlap.aux.update(act.aux)
            overlap.mods.update(act.mods)
            overlap.subjects.update(act.subjects)
            overlap.objects.update(act.objects)
            overlap.pds.update(act.pds)
            overlap.is_neg = act.is_neg or overlap.is_neg
            overlap.parallel = act.parallel or overlap.parallel
            overlap.optional = act.optional or overlap.optional
        if idx in self.activities.keys() and len(potential_overlaps) > 0:
            self.activities.pop(idx)

    def merge_aux(self, act, idx):
        verbs = act.verbs
        potential_verbs = [verb_i for verb_i, verb in self.activities.items() if set(verbs) & set(verb.aux)]
        for verb_i in potential_verbs:
            verb = self.activities[verb_i]
            verb.aux.add(idx)
            verb.mods.update(act.mods)
            verb.subjects.update(act.subjects)
            verb.objects.update(act.objects)
            verb.pds.update(act.pds)
            verb.is_neg = act.is_neg or verb.is_neg
            verb.parallel = act.parallel or verb.parallel
            verb.optional = act.optional or verb.optional
        if len(potential_verbs) > 0:
            self.activities.pop(idx)

    def check_and_set_conjunction(self, cj1, cj2, doc):
        if USE_CONJ:
            temp = False
            for child in doc[cj2].children:
                if child.lemma_ in TMP_AFTER or \
                        bool(set([child_.lemma_ for child_ in child.children]) & set(TMP_AFTER)
                             and child.pos_ == 'VERB'):
                    temp = True
            if not temp:
                self.relations[(cj1, cj2)] = "conj"
        elif set([child.lemma_.lower() for child in doc[cj2].children]) & set(TMPS_PARALLEL):
            self.relations[(cj1, cj2)] = "conj"

    def check_and_set_parallel(self, children, idx):
        if set(children) & set(TMPS_PARALLEL):
            self.activities[idx].set_parallel()

    def check_and_set_optional(self, children, idx):
        if set(children) & set(MARKERS_EXISTS):
            self.activities[idx].set_optional()

    def get_conjuncts(self, act: Activity):
        tmp_subjects = act.subjects.copy()
        tmp_objects = act.objects.copy()
        # check activity subjects for elements included in chunks
        for sb in act.subjects:
            cj_fits = [cj1 for cj1, cj2 in self.conjuncts if sb == cj2 or cj2 in tmp_subjects] + \
                      [cj2 for cj1, cj2 in self.conjuncts if sb == cj1 or cj1 in tmp_subjects]
            tmp_subjects.update(cj_fits)

        act.subjects.update(tmp_subjects)

        for ob in act.objects:
            cj_fits = [cj1 for cj1, cj2 in self.conjuncts if ob == cj2 or cj2 in tmp_objects] + \
                      [cj2 for cj1, cj2 in self.conjuncts if ob == cj1 or cj1 in tmp_objects]
            tmp_objects.update(cj_fits)

        act.objects.update(tmp_objects)

    def get_matches(self, doc):
        matches = self.matcher(doc)
        return matches

    @abstractmethod
    def routine(self):
        pass
