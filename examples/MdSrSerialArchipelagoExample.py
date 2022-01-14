
# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np

from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.serial_archipelago import SerialArchipelago
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island

from bingo.local_optimizers.continuous_local_opt_md import ContinuousLocalOptimizationMD
from bingo.symbolic_regression.agraphMD.component_generator_md import ComponentGeneratorMD
from bingo.symbolic_regression.agraphMD.crossover_md import AGraphCrossoverMD
from bingo.symbolic_regression.agraphMD.generator_md import AGraphGeneratorMD
from bingo.symbolic_regression.agraphMD.mutation_md import AGraphMutationMD
from bingo.symbolic_regression.explicit_regression_md import ExplicitTrainingDataMD, ExplicitRegressionMD

POP_SIZE = 100
STACK_SIZE = 15


def equation_eval(x):
    mu = 2.0
    alpha = 1.5
    C = np.array([[2.0 * mu + alpha, alpha, 0.0],
                  [alpha, 2.0 * mu + alpha, 0.0],
                  [0.0, 0.0, mu]])
    return C @ x[0] + C @ x[1]


def execute_generational_steps():
    n_points = 100
    x_0 = np.linspace(0, 11, 3 * n_points).reshape((-1, 3, 1))
    x_1 = np.linspace(-10, 0, 3 * n_points).reshape((-1, 3, 1))
    x = [x_0, x_1]
    y = equation_eval(x)
    training_data = ExplicitTrainingDataMD(x, y)

    x_dims = [np.shape(_x[0]) for _x in x]

    component_generator = ComponentGeneratorMD(x_dims)
    component_generator.add_operator("+")
    component_generator.add_operator("*")

    crossover = AGraphCrossoverMD()
    mutation = AGraphMutationMD(component_generator, command_probability=0.333, node_probability=0.333,
                                parameter_probability=0.333, prune_probability=0.0, fork_probability=0.0)
    agraph_generator = AGraphGeneratorMD(STACK_SIZE, component_generator, x_dims, y[0].shape, use_simplification=False)

    fitness = ExplicitRegressionMD(training_data=training_data)
    local_opt_fitness = ContinuousLocalOptimizationMD(fitness, algorithm='lm', param_init_bounds=[0, 0])
    evaluator = Evaluation(local_opt_fitness)

    ea = AgeFitnessEA(evaluator, agraph_generator, crossover,
                      mutation, 0.4, 0.4, POP_SIZE)

    island = Island(ea, agraph_generator, POP_SIZE)
    archipelago = SerialArchipelago(island)

    opt_result = archipelago.evolve_until_convergence(max_generations=500,
                                                      fitness_threshold=1.0e-4)

    print(opt_result)
    best_indiv = archipelago.get_best_individual()
    print(best_indiv.get_formatted_string("console"))
    print(best_indiv.command_array)
    print(best_indiv.constants)
    print(best_indiv.fitness)


def main():
    execute_generational_steps()


if __name__ == '__main__':
    main()
