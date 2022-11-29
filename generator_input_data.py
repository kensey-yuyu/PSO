""" This program is generating the text file of initial positions with random.
    You're able to change parameters of the number of particles, dimensions
    and the number of initial position limit. 
"""

import numpy as np


# parameters
# [N] is the number of particles.
# [DIMENSION] is the number of dimensions of function.
# [INITIAL_POSITION_MAX] is the position limit of initial speed. (default 100)
N = 30
DIMENSION = 20
POSITION_MAX = 100

position = np.random.uniform(-POSITION_MAX,
                             POSITION_MAX, (N, DIMENSION))
string_input = ""
for p in position:
    string_input += " ".join(map(str, p))
    string_input += "\n"
f = open("input_data.txt", "x", encoding="UTF-8")
f.write(string_input)
f.close
