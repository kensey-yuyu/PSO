""" This program is PSO with input initial position data.
    You're able to change parameters, initial position and function of evaluation. 
    First, run with "./input_data.txt".
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import function


# parameters
# [ITERATIONS] is the number of iteration of updating particle's position.
# [N] is the number of particles.
# [DIMENSION] is the number of dimensions of function.i
# [INITIAL_SPEED_MAX] is the speed limit of initial speed. (default 0)
# [WEIGHT], [C1], [C2] are the parameters of updating speed.
ITERATIONS = 10000
N = 30
DIMENSION = 20
INITIAL_SPEED_MAX = 0
WEIGHT = 0.729
C1 = 1.4955
C2 = 1.4955


def main():
    # init.
    data = input_data()
    generations = []
    particles = create_particles(data)
    particles = evaluate(particles)
    personal_best, global_best = [], []
    personal_best = update_personal_best(personal_best, particles)
    global_best = update_global_best(global_best, personal_best)

    # analysis.
    for _ in range(ITERATIONS):
        particles = update_speed(particles, personal_best, global_best)
        particles = update_position(particles)
        particles = evaluate(particles)
        personal_best = update_personal_best(personal_best, particles)
        global_best = update_global_best(global_best, personal_best)
        generations.append(copy.deepcopy(particles))

    # display answer.
    print("Answer:\n" + str(global_best[-1].position))
    print("fitness: " + str(global_best[-1].fitness))
    plot(global_best)
    return


class Particle:
    def __init__(self, d):
        self.position = np.array([d])
        self.speed = np.random.uniform(-INITIAL_SPEED_MAX,
                                       INITIAL_SPEED_MAX, (1, DIMENSION))
        self.fitness = -1
        return

    def get_fitness(self):
        return self.fitness


def input_data():
    file_name = input("Enter a file name of input data.\n")
    try:
        f = open(file_name, "r", encoding="UTF-8")
    except OSError as e:
        print(e)
    else:
        txt_data = f.readlines()
        data = []
        for d in txt_data:
            tmp = []
            for i in d.strip("\n").split(" "):
                tmp.append(float(i))
            tmp = np.array(tmp)
            data.append(tmp)
        data = np.array(data)
        f.close()
        return data


def create_particles(input):
    particles = []
    for i in input:
        particles.append(Particle(i))
    return particles


def evaluate(particles):
    # Select the function of evaluation on "function.py".
    return function.sphere_function(particles, N)


def update_personal_best(personal_best, particles):
    if personal_best == []:
        personal_best = copy.deepcopy(particles)
    else:
        for i in range(N):
            if particles[i].fitness < personal_best[i].fitness:
                personal_best[i] = copy.deepcopy(particles[i])
    return personal_best


def update_global_best(global_best, personal_best):
    global_best.append(copy.deepcopy(
        min(personal_best, key=Particle.get_fitness)))
    return global_best


def update_speed(particles, personal_best, global_best):
    for i in range(N):
        for j in range(DIMENSION):
            particles[i].speed[0][j] = WEIGHT * particles[i].speed[0][j] \
                + C1 * np.random.rand() * (global_best[-1].position[0][j] - particles[i].position[0][j]) \
                + C2 * np.random.rand() * \
                (personal_best[i].position[0]
                 [j] - particles[i].position[0][j])
    return particles


def update_position(particles):
    for i in range(N):
        particles[i].position += particles[i].speed
    return particles


def plot(global_best):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()
    x = []
    y = []
    for gen in range(ITERATIONS):
        x.append(gen)
        y.append(global_best[gen].fitness)
    ax1.plot(x, y)
    ax1.set_yscale("log")
    ax1.set_title("The relationship between Iteration and Evaluation value")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Evaluation value")
    plt.show()
    return


if __name__ == "__main__":
    main()
