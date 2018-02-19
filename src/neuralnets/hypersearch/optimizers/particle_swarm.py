import numpy as np
from numpy.random import rand
from collections import OrderedDict
import timeit
from collections import namedtuple

from search_space import param_decorator, SearchSpace
from src.neuralnets.hypersearch.optimizers.utils import generate_sobol_sequences


class Particle:
    def __init__(self, position, phi1, phi2, lb, ub):
        self.phi1 = phi1
        self.phi2 = phi2
        self.lb = lb
        self.ub = ub

        self.velocity = np.ones(position.shape)
        self.position = position

        self.best_position = self.position
        self.best_fitness = np.inf

    def get_position(self):
        return self.position

    def update_position(self, global_best_position):
        v1 = np.random.uniform(0, self.phi1, len(self.position)) * (self.best_position - self.position)
        v2 = np.random.uniform(0, self.phi2, len(self.position)) * (global_best_position - self.position)

        self.velocity = self.velocity + v1 + v2

        self.velocity = np.minimum(self.velocity, self.ub - self.position)
        self.velocity = np.maximum(self.velocity, self.lb - self.position)

        self.position = self.position + self.velocity
        return self.position

    def evaluate_fitness(self, f):
        fitness = f(*self.position)

        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position
        return fitness


class PSO:
    def __init__(self, num_generations, num_particles, lb, ub, phi1=1.5, phi2=2.0):
        self.num_generations = num_generations
        self.num_particles = num_particles

        positions = generate_sobol_sequences(num_particles, lb, ub)
        self.swarm = [Particle(pos, phi1, phi2, lb, ub) for pos in positions]

        self.best_fitness = np.inf
        self.best_position = rand(len(lb))

    def optimize(self, eval_fn):

        for i in range(self.num_generations):
            for j, particle in enumerate(self.swarm):

                # update particle
                particle.update_position(self.best_position)

                # evaluate particle
                fitness = particle.evaluate_fitness(eval_fn)

                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_position = particle.get_position()

                self.swarm[j] = particle

        return self.best_position, self.best_fitness


class PSOptimizer:
    def __init__(self, num_generations, num_particles, phi1=1.5, phi2=2.0):
        self.num_generations = num_generations
        self.num_particles = num_particles
        self.phi1 = phi1
        self.phi2 = phi2

    def optimize(self, run_f, params):
        search_tree = SearchSpace(params)

        lb = search_tree.get_lb()
        ub = search_tree.get_ub()
        f = param_decorator(run_f, search_tree)

        pso = PSO(self.num_generations, self.num_particles, lb, ub, self.phi1, self.phi2)

        start = timeit.default_timer()
        best_params, score = pso.optimize(f)
        end = timeit.default_timer() - start

        best_params = search_tree.transform(best_params)
        Result = namedtuple('Result', ['params', 'score', 'time'])

        return Result(best_params, score, end)


def func(x, y):
    return x ** 2 + y ** 2


def main():
    params = OrderedDict()
    params['x'] = (int, -100, 100)
    params['y'] = (int, 1, 100)

    opt = PSOptimizer(10, 10)
    res = opt.optimize(func, params)
    print res.params
    print res.score
    print res.time


if __name__ == '__main__':
    main()
