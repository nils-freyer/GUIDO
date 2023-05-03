from typing import Set, Tuple, List, Dict

import pandas as pd


def sort_activities(act_ids: List[int], rev_tuples: Set[Tuple[int, int]], i2act: Dict) -> List[int]:
    act_ids = sorted(act_ids)
    out_list = act_ids.copy()
    for i in range(len(act_ids) - 1):
        if (act_ids[i], act_ids[i + 1]) in rev_tuples:
            out_list[i] = act_ids[i + 1]
            out_list[i + 1] = act_ids[i]
            x = 0
        if i2act[act_ids[i]].Negative:
            out_list.pop(i)
    return out_list


def filter_negatives(sent2acts: Dict, i2act: Dict) -> Dict:
    for sent_id, act_ids in sent2acts.items():
        out_act_ids = act_ids.copy()
        for act_id in act_ids:
            if i2act[act_id].Negative:
                out_act_ids.remove(act_id)
        sent2acts[sent_id] = out_act_ids
    return sent2acts


def transitive_closure(a: set):
    closure = set(a)
    while True:
        new_relations = set((x, w) for x, y in closure for q, w in closure if q == y)

        closure_until_now = closure | new_relations

        if closure_until_now == closure:
            break

        closure = closure_until_now

    return closure


def sym_closure(a: set):
    closure = set(a)
    for (e, e_) in a:
        closure.add((e_, e))
    return closure


def filter_closure(closure: set, activities):
    closure = set([a for a in closure if a[0] in activities.id.values and a[1] in activities.id.values])
    return closure


def get_ccjs(act_id: int, cj_closure: set):
    conjuncts = {act_id}
    for cj in cj_closure:
        if cj[0] == act_id:
            conjuncts.add(cj[1])
        if cj[1] == act_id:
            conjuncts.add(cj[0])
    return conjuncts


def get_parallel_conjs(sent2acts: Dict, i2act: Dict) -> Set[Tuple[int, int]]:
    parallel_conjunction_tuples = set()
    for act_ids in sent2acts.values():
        for i in range(1, len(act_ids)):
            if i2act[act_ids[i]].Parallel:
                parallel_conjunction_tuples.add((act_ids[i], act_ids[i - 1]))
    return parallel_conjunction_tuples


def read_extracted_data(file):
    constraints = pd.read_json(f'output/{file}_constraints.jsonl', lines=True)
    activities = pd.read_json(f'output/{file}_activities.jsonl', lines=True)
    relations = pd.read_json(f'output/{file}_relations.jsonl', lines=True)
    return activities, constraints, relations


def get_parallels(sent2net: Dict, sent2acts: Dict, i2act: Dict) -> Set[Tuple[int, int]]:
    parallels = set()
    for idx in range(1, len(sent2net)):
        ids = list(sent2acts.keys())
        curr_sent_id = ids[idx]
        last_sent_id = ids[idx - 1]
        act_ids = sent2acts[curr_sent_id]
        if len(act_ids) > 0:
            if i2act[act_ids[0]].Parallel:
                parallels.add((last_sent_id, curr_sent_id))

    return parallels


def get_parallels_by_id(sent_id: int, parallels: Set[Tuple[int, int]]):
    for p in parallels:
        if sent_id in p:
            return p
