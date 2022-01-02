import numpy as np

from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.serial_archipelago import SerialArchipelago
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island

from bingo.local_optimizers.continuous_local_opt_md import ContinuousLocalOptimizationMD
from bingo.symbolic_regression import AGraphCrossover
from bingo.symbolic_regression.agraphMD.agraphMD import AGraphMD
from bingo.symbolic_regression.agraphMD.component_generator_md import ComponentGeneratorMD
from bingo.symbolic_regression.agraphMD.generator_md import AGraphGeneratorMD
from bingo.symbolic_regression.agraphMD.mutation_md import AGraphMutationMD
from bingo.symbolic_regression.agraphMD.operator_definitions import *
from bingo.symbolic_regression.explicit_regression_md import ExplicitTrainingDataMD, ExplicitRegressionMD

POP_SIZE = 100
STACK_SIZE = 10


def get_sigma(epsilon):
    P_sym = np.array([[1, 0, 0, 0],
                      [0, 1/2, 1/2, 0],
                      [0, 1/2, 1/2, 0],
                      [0, 0, 0, 1]])
    P_skw = np.array([[0, 0, 0, 0],
                      [0, 1/2, -1/2, 0],
                      [0, -1/2, 1/2, 0],
                      [0, 0, 0, 0]])

    M_sym = np.array([[1, 0, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 2, 0]])
    M_skw = np.array([[0, 2, 0, 0]])

    P_hat_sym = M_sym @ P_sym
    P_hat_skw = M_skw @ P_skw

    lambda_e = -120.74
    mu_e = 557.11
    mu_star_e = 8.37
    mu_c = 1.8e-4
    C_e = np.array([[2 * mu_e + lambda_e, lambda_e, 0],
                    [lambda_e, 2 * mu_e + lambda_e, 0],
                    [0, 0, mu_star_e]])
    C_c = np.array([[mu_c]])

    sigma = P_hat_sym.T @ C_e @ P_hat_sym @ epsilon + P_hat_skw.T @ C_c @ P_hat_skw @ epsilon
    print(P_hat_sym.T @ C_e @ P_hat_sym + P_hat_skw.T @ C_c @ P_hat_skw)
    return sigma


if __name__ == '__main__':
    # x = np.array([
    #     [[[0, 1, 2, 3]]],
    #     [[[4, 5, 6, 7]]]
    # ])
    # x = x.transpose((0, 1, 3, 2))  # make [0, 1, 2, 3], etc. column vectors
    x = np.linspace(-100, 101, 4*10).reshape((-1, 1, 4, 1))
    epsilon = x[:, 0]
    sigma = get_sigma(epsilon)

    training_data = ExplicitTrainingDataMD(x, sigma)

    print([np.shape(x_) for x_ in x[0]])
    print(sigma[0].shape)

    component_generator = ComponentGeneratorMD([np.shape(x_) for x_ in x[0]])
    component_generator.add_operator("+")
    component_generator.add_operator("*")

    crossover = AGraphCrossover()
    mutation = AGraphMutationMD(component_generator, command_probability=0.333, node_probability=0.333,
                                parameter_probability=0.333, prune_probability=0.0, fork_probability=0.0)
    agraph_generator = AGraphGeneratorMD(STACK_SIZE, component_generator, sigma[0].shape, use_simplification=False)

    fitness = ExplicitRegressionMD(training_data=training_data, metric="rmse")
    local_opt_fitness = ContinuousLocalOptimizationMD(fitness, algorithm="lm", param_init_bounds=[0, 0])
    evaluator = Evaluation(local_opt_fitness)

    ea = AgeFitnessEA(evaluator, agraph_generator, crossover,
                      mutation, 0.0, 0.2, POP_SIZE)

    island = Island(ea, agraph_generator, POP_SIZE)
    archipelago = SerialArchipelago(island)

    opt_result = archipelago.evolve_until_convergence(max_generations=500,
                                                      fitness_threshold=1.0e-10)

    print(opt_result)
    best_indiv = archipelago.get_best_individual()
    print(best_indiv.get_formatted_string("console"))
    print(best_indiv.command_array)
    print(best_indiv.constants)
    print(best_indiv.fitness)

    # testing clo on ideal form
    # test = AGraphMD(output_dim=(4, 1))
    # test.command_array = np.array([[VARIABLE, 0, 4, 1],
    #                                [CONSTANT, 0, 4, 4],
    #                                [MULTIPLICATION, 1, 0, 0]])
    # test._update()
    # local_opt_fitness(test)
    # print(test)
    # print(test.constants)
    # print(fitness(test))
