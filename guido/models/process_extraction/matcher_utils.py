CONJ = ['sowie', 'plus', 'außerdem', 'daneben', 'überdies', 'zugleich', 'zusätzlich', 'und', 'zudem']

MARKERS = ['können', 'dürfen', 'mögen', 'sollten', 'müssen', 'kann', ]
MARKERS_ALL = ['müssen']
MARKERS_EXISTS = ['können', 'dürfen', 'mögen', 'sollten', 'kann']

TMPS_PARALLEL = ['inzwischen', 'dabei', 'währenddessen', 'dazwischen', 'inzwischen', 'mittlerweile', 'solange',
                 'zwischenzeitlich', 'derweil', 'einstweilen']
TMP_BEFORE = ['zuvor', 'davor', 'vorab', 'vordem', 'vorher', 'vorweg', 'zuerst', 'zunächst', 'anfänglich', 'anfangs',
              'eingangs', 'erst', 'vorerst', ]
TMP_AFTER = ['schließlich', 'anschließend', 'danach', 'dann', 'hiernach', 'hinterher', 'nachfolgend', 'sodann', ]
# Spacy Matcher Patterns

V_PATTERN_CVC = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}, },
    {"LEFT_ID": "verb", "REL_OP": ">>", "RIGHT_ID": "cvc", "RIGHT_ATTRS": {"DEP": "cvc"}, },
    {"LEFT_ID": "cvc", "REL_OP": ">>", "RIGHT_ID": "nk", "RIGHT_ATTRS": {"DEP": "nk"}, },
]
V_PATTERN_OA = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}, },
    {"LEFT_ID": "verb", "REL_OP": ">>", "RIGHT_ID": "oa", "RIGHT_ATTRS": {"DEP": "oa"}, },
]

V_PATTERN_OC = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}, },
    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "oc", "RIGHT_ATTRS": {"DEP": "oc"}, },
]

V_PATTERN_OG = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}, },
    {"LEFT_ID": "verb", "REL_OP": ">>", "RIGHT_ID": "og", "RIGHT_ATTRS": {"DEP": "og"}, },
]

V_PATTERN_OP = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}, },
    {"LEFT_ID": "verb", "REL_OP": ">>", "RIGHT_ID": "op", "RIGHT_ATTRS": {"DEP": "op"}, },
]

V_PATTERN_SB = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}, },
    {"LEFT_ID": "verb", "REL_OP": ">>", "RIGHT_ID": "sb", "RIGHT_ATTRS": {"DEP": "sb"}, },
]

V_PATTERN_SBP = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}, },
    {"LEFT_ID": "verb", "REL_OP": ">>", "RIGHT_ID": "sbp", "RIGHT_ATTRS": {"DEP": "sbp"}, },
]

V_PATTERN_MO = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}, },
    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "mo", "RIGHT_ATTRS": {"DEP": "mo"}, },
]

V_PATTERN_CJ = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}, },
    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "cj", "RIGHT_ATTRS": {"DEP": "cj", "POS": "VERB"}, },
]
V_PATTERN_CJ_CD = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}, },
    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "cd", "RIGHT_ATTRS": {"DEP": "cd"}, },
    {"LEFT_ID": "cd", "REL_OP": ">", "RIGHT_ID": "cj", "RIGHT_ATTRS": {"DEP": "cj"}, },
]

V_PATTERN_NEG = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}, },
    {"LEFT_ID": "verb", "REL_OP": ">>", "RIGHT_ID": "ng", "RIGHT_ATTRS": {"DEP": "ng"}, },
]

A_PATTERN_OA = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "AUX"}, },
    {"LEFT_ID": "verb", "REL_OP": ">>", "RIGHT_ID": "oa", "RIGHT_ATTRS": {"DEP": "oa"}, },
]

A_PATTERN_OC = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "AUX"}, },
    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "oc", "RIGHT_ATTRS": {"DEP": "oc"}, },
]

A_PATTERN_OG = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "AUX"}, },
    {"LEFT_ID": "verb", "REL_OP": ">>", "RIGHT_ID": "og", "RIGHT_ATTRS": {"DEP": "og"}, },
]

A_PATTERN_OP = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "AUX"}, },
    {"LEFT_ID": "verb", "REL_OP": ">>", "RIGHT_ID": "op", "RIGHT_ATTRS": {"DEP": "op"}, },
]

A_PATTERN_SB = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "AUX"}, },
    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "sb", "RIGHT_ATTRS": {"DEP": "sb"}, },
]

A_PATTERN_SBP = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "AUX"}, },
    {"LEFT_ID": "verb", "REL_OP": ">>", "RIGHT_ID": "sbp", "RIGHT_ATTRS": {"DEP": "sbp"}, },
]

A_PATTERN_VERB = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "VERB"}, },
    {"LEFT_ID": "verb", "REL_OP": "<<", "RIGHT_ID": "aux", "RIGHT_ATTRS": {"POS": "AUX"}, },
]

A_PATTERN_NEG = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "AUX"}, },
    {"LEFT_ID": "verb", "REL_OP": ">>", "RIGHT_ID": "ng", "RIGHT_ATTRS": {"DEP": "ng"}, },
]

A_PATTERN_CJ = [
    {"RIGHT_ID": "aux", "RIGHT_ATTRS": {"POS": "AUX"}, },
    {"LEFT_ID": "aux", "REL_OP": ">>", "RIGHT_ID": "cj", "RIGHT_ATTRS": {"DEP": "cj", "POS": "AUX"}, },
]

A_PATTERN_PD = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "AUX"}, },
    {"LEFT_ID": "verb", "REL_OP": ">>", "RIGHT_ID": "pd", "RIGHT_ATTRS": {"DEP": "pd"}, },
]

A_PATTERN_MO = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "AUX"}, },
    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "mo", "RIGHT_ATTRS": {"DEP": "mo"}, },
]

N_PATTERN_AG_MO = [
    {"RIGHT_ID": "noun", "RIGHT_ATTRS": {"POS": "NOUN"}, },
    {"LEFT_ID": "noun", "REL_OP": ">>", "RIGHT_ID": "ag", "RIGHT_ATTRS": {"DEP": "ag"}, },
    {"LEFT_ID": "noun", "REL_OP": "<", "RIGHT_ID": "mo", "RIGHT_ATTRS": {"DEP": "mo"}, },
]

N_PATTERN_AG = [
    {"RIGHT_ID": "noun", "RIGHT_ATTRS": {"POS": "NOUN"}, },
    {"LEFT_ID": "noun", "REL_OP": ">>", "RIGHT_ID": "ag", "RIGHT_ATTRS": {"DEP": "ag"}, },
]

N_PATTERN_PG = [
    {"RIGHT_ID": "noun", "RIGHT_ATTRS": {"POS": "NOUN"}, },
    {"LEFT_ID": "noun", "REL_OP": ">>", "RIGHT_ID": "pg", "RIGHT_ATTRS": {"DEP": "pg"}, },
    {"LEFT_ID": "pg", "REL_OP": ">>", "RIGHT_ID": "nk", "RIGHT_ATTRS": {"DEP": "nk"}, },
]

N_PATTERN_CJ = [
    {"RIGHT_ID": "noun", "RIGHT_ATTRS": {"POS": "NOUN"}, },
    {"LEFT_ID": "noun", "REL_OP": ">", "RIGHT_ID": "cj", "RIGHT_ATTRS": {"DEP": "cj"}, },
]

N_PATTERN_CJ_CD = [
    {"RIGHT_ID": "noun", "RIGHT_ATTRS": {"POS": "NOUN"}, },
    {"LEFT_ID": "noun", "REL_OP": ">", "RIGHT_ID": "cd", "RIGHT_ATTRS": {"DEP": "cd"}, },
    {"LEFT_ID": "cd", "REL_OP": ">", "RIGHT_ID": "cj", "RIGHT_ATTRS": {"DEP": "cj"}, },
]

PN_PATTERN_CJ = [
    {"RIGHT_ID": "noun", "RIGHT_ATTRS": {"POS": "PROPN"}, },
    {"LEFT_ID": "noun", "REL_OP": ">>", "RIGHT_ID": "mo", "RIGHT_ATTRS": {"DEP": "cj"}, },
]
