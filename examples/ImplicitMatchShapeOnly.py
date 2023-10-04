import numpy as np

from bingo.symbolic_regression.implicit_regression import ImplicitRegression, ImplicitTrainingData
from bingo.symbolic_regression.agraph.agraph import AGraph

from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.serial_archipelago import SerialArchipelago
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.local_opt_fitness \
    import LocalOptFitnessFunction

from bingo.symbolic_regression import ComponentGenerator, \
    AGraphGenerator, \
    AGraphCrossover, \
    AGraphMutation

POP_SIZE = 200
STACK_SIZE = 10


def get_ideal_eq():
    # TODO these shouldn't give different results!!!!
    # ideal_eq = AGraph(equation="X_0**2 + X_1**2")
    ideal_eq = AGraph(equation="X_0*X_0 + X_1*X_1")
    return ideal_eq


def get_circle_data():
    t = np.linspace(0, 2 * np.pi, num=30)
    radii = np.linspace(1, 5, num=5)
    circle_points = np.empty((radii.shape[0], t.shape[0], 2))
    for i, radius in enumerate(radii):
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        points = np.hstack((x[:, None], y[:, None]))
        circle_points[i] = points

    # for non-transposed version (match expansion instead of shape)
    # circle_points = circle_points.transpose((1, 0, 2))

    implicit_data = []
    for circle_point_set in circle_points:
        implicit_data.extend(circle_point_set)
        implicit_data.append(np.full(2, np.nan))
    implicit_data = np.array(implicit_data[:-1])
    return implicit_data


def main():
    circle_points = get_circle_data()
    x = circle_points
    print(x.shape)

    training_data = ImplicitTrainingData(x)
    x = training_data._x

    fitness = ImplicitRegression(training_data)

    optimizer = ScipyOptimizer(fitness, method="BFGS", param_init_bounds=[-1, 1])
    local_opt_fitness = LocalOptFitnessFunction(fitness, optimizer)

    ideal_eq = get_ideal_eq()
    print("fitness of ideal_eq:", local_opt_fitness(ideal_eq))

    component_generator = ComponentGenerator(x.shape[1])
    component_generator.add_operator("+")
    component_generator.add_operator("-")
    component_generator.add_operator("*")

    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)

    agraph_generator = AGraphGenerator(STACK_SIZE, component_generator,
                                       use_simplification=True)

    optimizer = ScipyOptimizer(fitness, method='lm')
    local_opt_fitness = LocalOptFitnessFunction(fitness, optimizer)
    evaluator = Evaluation(local_opt_fitness)

    ea = AgeFitnessEA(evaluator, agraph_generator, crossover,
                      mutation, 0.4, 0.4, POP_SIZE)

    island = Island(ea, agraph_generator, POP_SIZE)
    archipelago = SerialArchipelago(island)

    opt_result = archipelago.evolve_until_convergence(max_generations=500,
                                                      fitness_threshold=1.0e-4)
    print("found individual:", archipelago.get_best_individual())


if __name__ == '__main__':
    main()
