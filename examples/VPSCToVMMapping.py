# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import time

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)

from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.serial_archipelago import SerialArchipelago
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island

from bingo.local_optimizers.continuous_local_opt_md import ContinuousLocalOptimizationMD
from bingo.symbolic_regression.agraphMD.agraphMD import AGraphMD
from bingo.symbolic_regression.agraphMD.component_generator_md import ComponentGeneratorMD
from bingo.symbolic_regression.agraphMD.crossover_md import AGraphCrossoverMD
from bingo.symbolic_regression.agraphMD.generator_md import AGraphGeneratorMD
from bingo.symbolic_regression.agraphMD.mutation_md import AGraphMutationMD
from bingo.symbolic_regression.explicit_regression_md import ExplicitTrainingDataMD, ExplicitRegressionMD

from bingo.symbolic_regression.mapping_fitness_md import MappingFitnessTrainingData, MappingFitness

POP_SIZE = 100
STACK_SIZE = 10


def get_ideal_eq():
    cmd_arr = np.array([[1, 0, 3, 3]])

    from bingo.symbolic_regression.agraphMD.agraphMD import AGraphMD
    ideal_eq = AGraphMD(input_dims=[(0, 0)], output_dim=(3, 3))
    ideal_eq.command_array = cmd_arr
    ideal_eq._update()
    constants = np.array([[-0.59910444,  0.28617222,  0.31293222],
                       [ 0.302744  , -0.623804  ,  0.32106   ],
                       [ 0.32928822,  0.32084422, -0.65013244]])
    ideal_eq.set_local_optimization_params(constants.flatten(), [(3, 3)])
    from bingo.symbolic_regression.agraphMD.validation_backend.validation_backend import validate_individual
    print(validate_individual(ideal_eq))
    return ideal_eq


def main(use_pytorch=False):
    np.set_printoptions(suppress=True)
    # df = pd.read_pickle("vpsc_evo_17_data_3d_points_transpose_implicit_format_mapping_df.pkl")
    # df = pd.read_pickle("vpsc_evo_0_data_3d_points_implicit_format_mapping_df.pkl")
    # df = pd.read_pickle("vpsc_evo_57_data_3d_points_implicit_format_mapping_df.pkl")
    df = pd.read_pickle("hill_w_hardening_mapping_df.pkl")

    def get_numpy_matrix(df, column):
        column_list = df[column]
        numpy_version = []
        for data_entry in column_list:
            numpy_version.append(data_entry)
        return np.array(numpy_version)

    state_parameters = [get_numpy_matrix(df, "eps").astype(np.float64)]
    P_actual = get_numpy_matrix(df, "P_actual").astype(np.float64)
    P_desired = get_numpy_matrix(df, "P_vm").astype(np.float64)
    # P_desired *= P_desired * (state_parameters[0] + 1)[:, None, None]

    x_dims = [(0, 0) if np.shape(x_[0]) == () else np.shape(x_[0]) for x_ in state_parameters]
    y_dim = P_desired[0].shape
    if use_pytorch:
        state_parameters = [torch.from_numpy(state_parameters[0]).double()]

    # TODO can convert this to a parent agraph fitness using explicit regression
    training_data = MappingFitnessTrainingData(state_parameters=state_parameters,
                                               P_actual=P_actual,
                                               P_desired=P_desired)

    print("Dimensions of X variables:", x_dims)
    print("Dimension of output:", y_dim)

    component_generator = ComponentGeneratorMD(x_dims, possible_dims=[(3, 3), (0, 0)])
    component_generator.add_operator("+")
    component_generator.add_operator("*")
    component_generator.add_operator("sin")
    component_generator.add_operator("cos")
    component_generator.add_operator("sqrt")

    crossover = AGraphCrossoverMD()
    mutation = AGraphMutationMD(component_generator)
    agraph_generator = AGraphGeneratorMD(STACK_SIZE, component_generator, x_dims, y_dim, use_simplification=False,
                                         use_pytorch=use_pytorch)

    fitness = MappingFitness(training_data=training_data)
    local_opt_fitness = ContinuousLocalOptimizationMD(fitness, algorithm='lm', param_init_bounds=[-1, 1], options={"factor": 0.1, "xtol": 1e-15, "ftol": 1e-15, "gtol": 1e-15})
    # local_opt_fitness = ContinuousLocalOptimizationMD(fitness, algorithm='lm', param_init_bounds=[-1, 1])

    ideal_eq = get_ideal_eq()
    print("ideal eq fitness before opt:", fitness(ideal_eq))
    ideal_eq._needs_opt = True
    print("ideal eq fitness after opt:", local_opt_fitness(ideal_eq))
    print("\t constants:", ideal_eq.constants)

    evaluator = Evaluation(local_opt_fitness, multiprocess=4)

    ea = AgeFitnessEA(evaluator, agraph_generator, crossover,
                      mutation, 0.4, 0.4, POP_SIZE)

    island = Island(ea, agraph_generator, POP_SIZE)
    # archipelago = SerialArchipelago(island)

    opt_result = island.evolve_until_convergence(max_generations=500,
                                                 fitness_threshold=1.0e-5)

    print(local_opt_fitness.eval_count)
    print(opt_result)
    best_indiv = island.get_best_individual()
    print(best_indiv.get_formatted_string("console"))
    print(best_indiv.fitness)


if __name__ == '__main__':
    main(use_pytorch=False)
