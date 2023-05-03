from itertools import chain, combinations

import numpy as np
import pm4py
from numpy.linalg import norm
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.petri_net.utils.check_soundness import check_sink_place_presence, check_source_place_presence

from guido.utils.logging_utils import get_custom_logger

EXCEPTION_WFN = "Error: petri-net is not a workflow net. Convert first."

logger = get_custom_logger('similarity')


def pre_set(place: PetriNet.Place) -> frozenset:
    return frozenset({arc.source.name for arc in place.in_arcs})


def post_set(place: PetriNet.Place) -> frozenset:
    return frozenset({arc.target.name for arc in place.out_arcs})


def check_powerset(transitions, net, place_s, n, mode='la'):
    """

    :param net: petri net
    :param place_s: source if mode == b, else sink of net
    :param n: transition
    :param transitions: N (set of transitions)
    :param mode: look ahead: la, look back: lb
    :return:
    """
    s = list(transitions)
    output = []
    for r in range(len(s) + 1):
        subset = set(chain.from_iterable(combinations(s, r)))
        if mode == 'la':
            if check_footprint_a(net, place_s, n, subset):
                output.append(subset)
        else:
            if check_footprint_b(net, place_s, n, subset):
                output.append(subset)
    return output


def get_source(net: PetriNet) -> (PetriNet.Place, PetriNet.Transition):
    pot_source = check_source_place_presence(net=net)

    tr_source = PetriNet.Transition('initial_transition')
    return pot_source, tr_source


def get_sink(net: PetriNet) -> (PetriNet.Place, PetriNet.Transition):
    pot_sink = check_sink_place_presence(net=net)


    tr_sink = PetriNet.Transition('final_transition')
    return pot_sink, tr_sink


def check_footprint_a(net, sink, n, M) -> bool:
    for place in net.places:
        if place != sink and n in pre_set(place) and post_set(place) == M:
            return True
    return False


def check_footprint_b(net, source, n, M) -> bool:
    for place in net.places:
        if place != source and n in post_set(place) and pre_set(place) == M:
            return True
    return False


def filter_set(F):
    out_set = list()
    seen = set()
    for term in F:
        if f'{term}' not in seen:
            out_set.append(term)
            seen.add(f'{term}')
    return out_set


def convert_to_cfp(net: PetriNet):
    """
    Algorithm by Mendling, implementation by me :--)
     - (why isn't this in pm4py?)
    :param net:
    :return: cfp vectors of petri net
    """
    if not pm4py.analysis.check_is_workflow_net(net):
        raise Exception(EXCEPTION_WFN)

    source, tr_source = get_source(net)
    sink, tr_sink = get_sink(net)

    N = {tr.name for tr in list(net.transitions)}

    F_la = set()
    F_lb = set()

    for n in N:
        F_la.update(filter_set(
            [(n, M) for M in check_powerset(transitions=N, net=net, place_s=sink, n=n) if len(M) > 0]
            + [(tr_source.name, post_set(source))]
            + [(t, frozenset({tr_sink.name})) for t in pre_set(sink)]))

        F_lb.update(filter_set(
            [(M, n) for M in check_powerset(transitions=N, net=net, place_s=sink, n=n, mode='lb') if len(M) > 0]
            + [(pre_set(sink), tr_sink.name)]
            + [(frozenset({tr_source.name}), t) for t in post_set(source)]))

    N = N | {tr_source.name, tr_sink.name}

    return N, F_la, F_lb


def cosine_sim(u, v):
    """
    Similarity = (A.B) / (||A||.||B||)

    :param u: A
    :param v: B
    :return: cosine similarity
    """
    return np.dot(u, v) / (norm(u) * norm(v))


def get_weights(cfp_a, cfp_b):
    terms = filter_set(cfp_a[0] | cfp_a[1] | cfp_a[2] | cfp_b[0] | cfp_b[1] | cfp_b[2])
    t2i = {str(term): i for i, term in enumerate(terms)}
    weights = {t2i[str(term)]: 1 for term in filter_set(cfp_a[0] | cfp_b[0])}
    for term in filter_set(cfp_a[1] | cfp_b[1]):
        index = t2i[str(term)]
        weights[index] = 1 / (max([2 ^ len(term[1]), 1]))
    for term in filter_set(cfp_a[2] | cfp_b[2]):
        index = t2i[str(term)]
        weights[index] = 1 / (max([2 ^ len(term[0]), 1]))
    return weights, terms, t2i


def cfp_vectorize(cfp, t2i, weights):
    # lambda indexing function
    v = np.zeros(len(weights))
    for term in list(cfp[0] | cfp[1] | cfp[2]):
        v[t2i[str(term)]] = weights[t2i[str(term)]]
    return v


def behavioural_similarity(net_a: PetriNet, net_b: PetriNet) -> float:
    """
    :param net_a:
    :param net_b:
    :return: cfp based similarity score for petri nets
    """
    # TODO: semantic activity mapping? or do we just use the activity index in doc?
    logger.info(f'convert {net_a.name} to cfp')

    cfp_a = convert_to_cfp(net_a)

    logger.info(f'convert {net_b.name} to cfp')

    cfp_b = convert_to_cfp(net_b)

    weights, terms, t2i = get_weights(cfp_a, cfp_b)

    logger.info('Generate CFP Vectors')
    v_cfp_a = cfp_vectorize(cfp_a, t2i, weights)
    v_cfp_b = cfp_vectorize(cfp_b, t2i, weights)

    sim = cosine_sim(v_cfp_a, v_cfp_b)
    logger.info(f"Similarity for {net_a.name} to gold: {sim}")
    return sim
