""" Define the function of evaluation.
    Input value [particles] is necessary.
    Return value [particles] is necessary.
    Update each particle's fitness with the function.
"""

import numpy as np


def sphere_function(particles, n):
    # sphere function
    for i in range(n):
        particles[i].fitness = np.sum(particles[i].position ** 2)
    return particles
