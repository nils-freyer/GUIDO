from typing import List, Dict, Set, Tuple

import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.objects.petri_net.utils.check_soundness import check_source_place_presence, check_sink_place_presence

from guido.process_model.utils import transitive_closure, sym_closure, filter_closure, sort_activities, get_ccjs, \
    read_extracted_data, get_parallel_conjs, get_parallels_by_id, get_parallels, filter_negatives


def make_net(file: str):
    activities, constraints, relations = read_extracted_data(file)
    i2act = {row.id: row for _, row in activities.iterrows()}
    sent2acts = {sid: [act_id for act_id in constraints[constraints.sent_id == sid].act_id.values] for sid in
                 constraints.sent_id.values}
    parallel_conjunction_tuples = get_parallel_conjs(sent2acts=sent2acts, i2act=i2act)
    sent2acts = filter_negatives(sent2acts=sent2acts, i2act=i2act)
    if len(relations) > 0:
        revs = relations[relations.type == 'rev']
        rev_tuples = set(zip(revs.lact.values, revs.ract.values))
        conjunctions = relations[relations.type == 'conj']
        disjunctions = relations[relations.type == 'disj']

        conj_tuples = set(zip(conjunctions.lact.values, conjunctions.ract.values))
        conj_tuples.update(parallel_conjunction_tuples)
        disj_tuples = set(zip(disjunctions.lact.values, disjunctions.ract.values))
    else:
        rev_tuples = set()
        conj_tuples = set()
        disj_tuples = set()
    sent2acts = {sid: sort_activities(act_ids=act_ids, rev_tuples=rev_tuples, i2act=i2act) for sid, act_ids in
                 sent2acts.items()}

    conj_closure = transitive_closure(sym_closure(filter_closure(conj_tuples, activities)))
    disj_closure = transitive_closure(sym_closure(filter_closure(disj_tuples, activities)))

    pn = get_petri_net(sent2acts=sent2acts, i2act=i2act,
                       conj_closure=conj_closure, disj_closure=disj_closure,
                       file=file)

    png_path = f"output/petri_nets/{file}_pn.png"
    pn_path = f"output/petri_nets/{file}_pn.pnml"
    final_marking = initial_marking = Marking()
    pm4py.save_vis_petri_net(pn, initial_marking, final_marking, png_path)
    pm4py.write_pnml(pn, initial_marking, final_marking, pn_path)


def add_ccj(act_id: int, arcs: Set, seen: Set, start_place: PetriNet.Place, net: PetriNet,
            i2act: Dict, conj_closure: Set, disj_closure: Set):
    conjuncts_u = get_ccjs(act_id, conj_closure).difference(seen)
    disjuncts_u = get_ccjs(act_id, disj_closure)
    end = PetriNet.Place(f'END_CCJ{act_id}')

    if len(disjuncts_u.difference(seen)) <= 1 and len(conjuncts_u.difference(seen)) <= 1:
        return seen, start_place

    net.places.add(end)
    last_place = end

    if len(disjuncts_u.difference(seen)) > 1:
        transitions = dict()
        for v in disjuncts_u.difference(seen):
            seen.add(v)
            tr_v = PetriNet.Transition(f'{v}', f'{v}')
            net.transitions.add(tr_v)
            transitions[v] = tr_v
            arcs.add((start_place, tr_v))

        for tr in transitions.values():
            arcs.add((tr, end))

    if len(conjuncts_u.difference(seen)) > 1:
        lp_tmps = []
        tr_start_and = PetriNet.Transition(f'AND_{act_id}')
        arcs.add((start_place, tr_start_and))
        net.transitions.add(tr_start_and)
        start_places = dict()
        for v in conjuncts_u.difference(seen):
            seen.add(v)
            place = PetriNet.Place(f'p_and_{v}')
            start_places[v] = place
            end_and = PetriNet.Place(f'p_end{v, act_id}')
            tr_v = PetriNet.Transition(f'{v}', f'{v}')

            net.transitions.add(tr_v)
            net.places.add(place)
            net.places.add(end_and)
            arcs.add((tr_start_and, place))
            arcs.add((place, tr_v))
            arcs.add((tr_v, end_and))
            lp_tmps.append(end_and)
        tr_and_end = PetriNet.Transition(f'AND_END_{act_id}')
        net.transitions.add(tr_and_end)
        for p in lp_tmps:
            arcs.add((p, tr_and_end))
        for v, place in start_places.items():
            if i2act[v].Optional:
                arcs.add((place, tr_and_end))
        arcs.add((tr_and_end, end))
    return seen, last_place


def get_sub_net(sent_id: int, act_ids: List[int], i2act: Dict, conj_closure: Set, disj_closure: Set):
    arcs = set()
    seen = set()
    net = PetriNet("recipe-net")
    if len(act_ids) == 0:
        return net
    source = PetriNet.Place(f"source_{sent_id}")
    sink = source
    net.places.add(sink)
    last_tr = None
    for act_id in act_ids:
        new_seen, sink = add_ccj(act_id=act_id, arcs=arcs, seen=seen, start_place=sink, net=net, i2act=i2act,
                                 conj_closure=conj_closure, disj_closure=disj_closure)
        seen.update(new_seen)
        if act_id not in seen:
            tr_u = PetriNet.Transition(f'{act_id}', f'{act_id}')
            net.transitions.add(tr_u)
            arcs.add((sink, tr_u))
            last_sink = sink
            sink = PetriNet.Place(f"{sent_id}_{act_id}")
            net.places.add(sink)
            arcs.add((tr_u, sink))
            if i2act[act_id].Optional and last_tr is not None:
                arcs.add((last_tr, sink))
            elif i2act[act_id].Optional and last_tr is None:
                tr_dummy = PetriNet.Transition(f'{act_id}_dummy')
                net.transitions.add(tr_dummy)
                arcs.add((last_sink, tr_dummy))
                arcs.add((tr_dummy, sink))

            last_tr = tr_u
            seen.add(act_id)

    for u, v in arcs:
        petri_utils.add_arc_from_to(u, v, net)

    return net


def get_sent2net(sent2acts: Dict, i2act: Dict, conj_closure: Set, disj_closure: Set):
    sent2net = dict()
    for sent_id, act_ids in sent2acts.items():
        subnet = get_sub_net(sent_id=sent_id, act_ids=act_ids, i2act=i2act,
                             conj_closure=conj_closure, disj_closure=disj_closure)
        sent2net[sent_id] = subnet
    return sent2net


# %%
def get_petri_net(sent2acts: Dict, i2act: Dict, conj_closure: Set, disj_closure: Set, file: str) -> PetriNet:
    sent2net = get_sent2net(sent2acts=sent2acts, i2act=i2act, conj_closure=conj_closure, disj_closure=disj_closure)
    net = PetriNet(file)
    seen = set()
    for sent_id in sent2net.keys():
        if sent_id in seen:
            continue
        parallel = get_parallels_by_id(sent_id, get_parallels(sent2net, sent2acts, i2act))
        if parallel is not None:
            net_1 = merge_parallel(parallel, seen, sent2net)

        else:
            seen.add(sent_id)
            net_1 = sent2net[sent_id]

        sink = check_sink_place_presence(net=net)
        if sink is not None:
            source_1 = check_source_place_presence(net=net_1)
            source_arcs = [arc for arc in net_1.arcs if arc.source == source_1]
            petri_utils.remove_place(net_1, source_1)
            petri_utils.merge(trgt=net, nets=[net_1])
            for arc in source_arcs:
                petri_utils.add_arc_from_to(fr=sink, to=arc.target, net=net)
        else:
            petri_utils.merge(trgt=net, nets=[net_1])
    return net


def merge_parallel(parallel: Tuple, seen: Set[int], sent2net: Dict):
    seen.update(set(parallel))
    net_1 = sent2net[parallel[0]]
    net_2 = sent2net[parallel[1]]
    parallel_source = PetriNet.Place(f"parallel_source_{parallel[0]}_{parallel[1]}")
    parallel_sink = PetriNet.Place(f"parallel_sink_{parallel[0]}_{parallel[1]}")
    parallel_tr = PetriNet.Transition(f"PARALLEL_{parallel[0]}_{parallel[1]}")
    parallel_end_tr = PetriNet.Transition(f"PARALLEL_END_{parallel[0]}_{parallel[1]}")
    source_1 = check_source_place_presence(net=net_1)
    sink_1 = check_sink_place_presence(net=net_1)
    source_2 = check_source_place_presence(net=net_2)
    sink_2 = check_sink_place_presence(net=net_2)
    petri_utils.merge(trgt=net_1, nets=[net_2])
    net_1.places.update({parallel_source, parallel_sink})
    net_1.transitions.update({parallel_tr, parallel_end_tr})
    petri_utils.add_arc_from_to(fr=parallel_source, to=parallel_tr, net=net_1)
    petri_utils.add_arc_from_to(fr=parallel_tr, to=source_1, net=net_1)
    petri_utils.add_arc_from_to(fr=parallel_tr, to=source_2, net=net_1)
    petri_utils.add_arc_from_to(fr=sink_1, to=parallel_end_tr, net=net_1)
    petri_utils.add_arc_from_to(fr=sink_2, to=parallel_end_tr, net=net_1)
    petri_utils.add_arc_from_to(fr=parallel_end_tr, to=parallel_sink, net=net_1)

    return net_1
