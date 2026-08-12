import random
import subprocess
from itertools import combinations

import matplotlib.pyplot as plt

from utils import (
    Instance,
    Job,
    Task,
    export_incompatibilities,
    export_to_dot_pairs,
    export_to_dot_separated,
    export_to_dot_sets,
)

inst = Instance("large")
inst.load(f"instances/{inst.name.lower()}.json")
print(f"There are {len(inst.jobs)} jobs")


# selected_tasks = [i for i in range(1, len(inst.jobs)+1)][::2]
selected_tasks = [
    74,
    76,
    81,
    85,
    87,
    89,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    109,
    110,
    111,
    113,
    114,
    115,
    116,
    117,
    120,
    121,
    122,
    123,
    124,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    134,
    135,
    136,
    137,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    149,
    150,
    151,
    152,
    155,
    159,
    161,
    163,
    164,
    165,
    166,
    167,
    169,
    170,
    171,
    172,
    174,
    175,
    176,
    177,
    179,
    180,
    181,
    182,
    184,
    185,
    188,
    189,
    191,
    192,
    194,
    195,
    196,
    197,
    198,
    199,
    200,
    206,
    221,
    248,
    286,
    300,
    319,
    322,
    330,
    336,
    338,
    350,
    358,
    404,
    416,
    418,
    419,
    438,
    441,
    446,
    458,
    464,
    466,
    488,
    493,
    495,
    497,
    506,
    535,
    539,
    540,
    566,
    621,
    633,
    635,
    662,
    698,
    729,
    757,
    766,
    794,
    806,
    919,
    963,
][::6]  # step window [20, 21] for solving the huge instance with Gurobi with window = 1
selected_tasks = [random.choice(inst.jobs[j].T) for j in inst.jobs]
selected_tasks = [
    6,
    12,
    69,
    102,
    114,
    120,
    154,
    156,
    163,
    214,
    220,
    288,
    300,
]  # step window [10,11] for solving the large instance with Gurobi with window = 1
selected_tasks = [
    6,
    12,
    22,
    39,
    94,
    102,
    114,
    156,
    158,
    189,
    196,
    288,
    298,
    300,
]  # step window [11,12] for solving the large instance with Gurobi with window = 1
selected_tasks = [
    120,
    58,
    209,
    99,
    71,
    155,
    12,
    288,
    102,
    114,
    198,
]  # step window [9,10] for solving the large instance with Gurobi with window = 1
print(f"There are {len(selected_tasks)} tasks selected")
selected_tasks = selected_tasks[::1]
print(f"{len(selected_tasks)} tasks were kept")

# Graph W
dot_filename = "./figures/graph_w.dot"
export_incompatibilities(inst, selected_tasks, dot_filename, display_tasks=False)
print(f"DOT file generated: {dot_filename}")

# generate the image using sfdp
img_filename = "./figures/graph_w.png"
subprocess.run(
    [
        "dot",
        "-Tpng",
        dot_filename,
        "-o",
        img_filename,
        "-Goverlap=false",
        "-Gnodesep=1.0",
        "-Granksep=0.8",
    ],
    check=False,
)


# Graph SCL
dot_filename = "./figures/graph_scl.dot"
# export_to_dot_networkx(inst, selected_tasks)

dot_filename = "./figures/graph_scl.dot"
export_to_dot_sets(inst, selected_tasks, dot_filename)
print(f"DOT file generated: {dot_filename}")

# generate the image using sfdp
img_filename = "./figures/graph_scl.png"
subprocess.run(
    [
        "sfdp",
        "-Tpng",
        dot_filename,
        "-o",
        img_filename,
        "-Goverlap=prism",
        "-Gsep=+60",
        "-GK=1",
        "-Gmaxiter=1000",
    ],
    check=True,
)


# Graph pair
dot_filename = "./figures/graph_pair.dot"
export_to_dot_pairs(inst, selected_tasks, dot_filename)
print(f"DOT file generated: {dot_filename}")

# generate the image using sfdp
img_filename = "./figures/graph_pair.png"
subprocess.run(
    [
        "dot",
        "-Tpng",
        dot_filename,
        "-o",
        img_filename,
        "-Goverlap=false",
        "-Gnodesep=1.0",
        "-Granksep=0.8",
    ],
    check=False,
)

# Graph sep
dot_filename = "./figures/graph_sep.dot"
export_to_dot_separated(inst, selected_tasks, dot_filename)
print(f"DOT file generated: {dot_filename}")

# generate the image using sfdp
img_filename = "./figures/graph_sep.png"
subprocess.run(
    [
        "dot",
        "-Tpng",
        dot_filename,
        "-o",
        img_filename,
        "-Goverlap=false",
        "-Gnodesep=1.0",
        "-Granksep=0.8",
    ],
    check=False,
)
