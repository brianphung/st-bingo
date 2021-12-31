
# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np

from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.serial_archipelago import SerialArchipelago
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island

from bingo.symbolic_regression import AGraphCrossover, \
                                      AGraphMutation

from bingo.local_optimizers.continuous_local_opt_md import ContinuousLocalOptimizationMD
from bingo.symbolic_regression.agraphMD.component_generator_md import ComponentGeneratorMD
from bingo.symbolic_regression.agraphMD.generator_md import AGraphGeneratorMD
from bingo.symbolic_regression.explicit_regression_md import ExplicitTrainingDataMD, ExplicitRegressionMD

POP_SIZE = 100
STACK_SIZE = 4


def init_x_vals(start, stop, num_points):
    return np.array([np.linspace(start, stop, num_points).reshape([3, -1])])


def equation_eval(x):
    mu = 2.0
    alpha = 1.5
    C = np.array([[2.0 * mu + alpha, alpha, 0.0],
                  [alpha, 2.0 * mu + alpha, 0.0],
                  [0.0, 0.0, mu]])
    return C @ x


def execute_generational_steps():
    np.random.seed(8)
    x = init_x_vals(-10, 10, 3)
    y = equation_eval(x)
    training_data = ExplicitTrainingDataMD(x, y)

    component_generator = ComponentGeneratorMD(x[0].shape)
    component_generator.add_operator("+")
    component_generator.add_operator("-")
    component_generator.add_operator("*")

    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)
    agraph_generator = AGraphGeneratorMD(STACK_SIZE, component_generator, y[0].shape, use_simplification=False)

    fitness = ExplicitRegressionMD(training_data=training_data)
    local_opt_fitness = ContinuousLocalOptimizationMD(fitness, algorithm='BFGS')
    evaluator = Evaluation(local_opt_fitness)

    ea = AgeFitnessEA(evaluator, agraph_generator, crossover,
                      mutation, 0.0, 0.0, POP_SIZE)

    island = Island(ea, agraph_generator, POP_SIZE)
    archipelago = SerialArchipelago(island)

    opt_result = archipelago.evolve_until_convergence(max_generations=500,
                                                      fitness_threshold=1.0e-4)
    if opt_result.success:
        best_indiv = archipelago.get_best_individual()
        print(best_indiv.get_formatted_string("console"))
        print(best_indiv.command_array)
        print(best_indiv.constants)
        print(best_indiv.fitness)
    else:
        print("Failed.")

    print(opt_result.ea_diagnostics)


def main():
    execute_generational_steps()


if __name__ == '__main__':
    main()
