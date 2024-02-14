import dill

import numpy as np
import torch
torch.set_num_threads(1)

from bingo.evaluation.fitness_function import VectorBasedFunction
from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.serial_archipelago import SerialArchipelago
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island
from bingo.stats.pareto_front import ParetoFront

from bingo.local_optimizers.continuous_local_opt_md import ContinuousLocalOptimizationMD
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.symbolic_regression.agraphMD.component_generator_md import ComponentGeneratorMD
from bingo.symbolic_regression.agraphMD.crossover_md import AGraphCrossoverMD
from bingo.symbolic_regression.agraphMD.generator_md import AGraphGeneratorMD
from bingo.symbolic_regression.agraphMD.mutation_md import AGraphMutationMD
from bingo.symbolic_regression.explicit_regression_md import ExplicitRegressionMD, ExplicitTrainingDataMD
from bingo.symbolic_regression.implicit_regression_md import ImplicitRegressionMD, ImplicitTrainingDataMD, \
    _calculate_partials
from bingo.symbolic_regression.implicit_regression_schmidt import ImplicitRegressionSchmidt

from bingo.symbolic_regression.agraphMD.validation_backend import validation_backend

POP_SIZE = 100
STACK_SIZE = 15


def get_ideal_eq_2():
    from bingo.symbolic_regression.agraphMD.pytorch_agraph_md import PytorchAGraphMD
    ideal_eq_cmd_arr = np.array([[1, 0, 3, 3],
                                 [0, 0, 0, 0],
                                 [1, 0, 0, 0],
                                 [4, 0, 1, 0],
                                 [4, 2, 3, 0]])
    ideal_eq = PytorchAGraphMD(input_dims=[(0, 0)], output_dim=(3, 3))
    ideal_eq.command_array = ideal_eq_cmd_arr
    ideal_eq._update()
    flattened_params = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 1])
    ideal_eq.set_local_optimization_params(flattened_params, [(3, 3), (0, 0)])
    print(validation_backend.validate_individual(ideal_eq))
    return ideal_eq


# def get_ideal_eq():
#     from bingo.symbolic_regression.agraphMD.pytorch_agraph_md import PytorchAGraphMD
#     ideal_eq_cmd_arr = np.array([[1, 0, 3, 3],
#                                  [2, 0, 0, 0],
#                                  [0, 0, 0, 0],
#                                  [4, 2, 2, 1],
#                                  [4, 3, 2, 1],
#                                  [2, 1, 1, 0],
#                                  [6, 2, 0, 0],
#                                  [4, 0, 6, 0],
#                                  [2, 6, 6, 0],
#                                  [4, 1, 3, 1],
#                                  [1, 1, 3, 3],
#                                  [7, 9, 9, 0],
#                                  [6, 11, 0, 0],
#                                  [4, 10, 12, 0],
#                                  [7, 13, 7, 0]])
#     ideal_eq = PytorchAGraphMD(input_dims=[(0, 0)], output_dim=(3, 3))
#     ideal_eq.command_array = ideal_eq_cmd_arr
#     ideal_eq._update()
#     flattened_params = np.hstack((np.array([[0.36211993, 0.58132813, 0.25392933],
#                                             [-0.59313177, -0.32607954, 0.45014156],
#                                             [-0.30666314, -0.28024253, -0.42051936]]).flatten(),
#                                   np.array([[0.08985267, 0.36115583, -0.46863242],
#                                             [-0.03694589, 0.00352989, -0.14291358],
#                                             [-0.46893991, -0.32886969, -0.98780778]]).flatten()))
#     ideal_eq.set_local_optimization_params(flattened_params, [(3, 3), (3, 3)])
#     print(validation_backend.validate_individual(ideal_eq))
#     return ideal_eq


def get_ideal_eq():
    cmd_arr = np.array([[1, 0, 3, 3],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [7, 2, 2, 1],
                        [4, 1, 3, 1],
                        [0, 0, 0, 0],
                        [4, 4, 5, 1],
                        [1, 1, 3, 3],
                        [4, 6, 7, 1],
                        [2, 0, 8, 0],
                        [6, 9, 8, 0],
                        [1, 2, 3, 3],
                        [4, 11, 11, 1],
                        [7, 12, 12, 1],
                        [4, 10, 13, 1]])
    from bingo.symbolic_regression.agraphMD.pytorch_agraph_md import PytorchAGraphMD
    ideal_eq = PytorchAGraphMD(input_dims=[(0, 0)], output_dim=(3, 3))
    ideal_eq.command_array = cmd_arr
    ideal_eq._update()

    params = (np.array([[-3.65976109, 12.33662161, -0.95124926],
                        [-3.47534396, 6.60816029, -6.85610072],
                        [-0.10874782, -1.90679975, 0.75421104]]),
              np.array([[7.59550467, -0.80455126, -6.50451482],
                        [-3.33794485, -5.29044806, -2.51050226],
                        [-36.94928943, -68.71260014, 7.63713728]]),
              np.array([[-0.81958222, -2.44316608, 0.10018877],
                        [0.29607395, -4.34619018, -2.61235277],
                        [-0.17601067, 0.14779934, -0.86227791]]))
    print(params)
    flattened_params = np.hstack([param.flatten() for param in params])
    ideal_eq.set_local_optimization_params(flattened_params, [(3, 3)] * len(params))
    print(validation_backend.validate_individual(ideal_eq))
    print(ideal_eq)
    return ideal_eq


def get_ideal_eq():
    cmd_arr = np.array([[1, 0, 3, 3]])
    from bingo.symbolic_regression.agraphMD.pytorch_agraph_md import PytorchAGraphMD
    ideal_eq = PytorchAGraphMD(input_dims=[(0, 0)], output_dim=(3, 3))
    ideal_eq.command_array = cmd_arr
    ideal_eq._update()
    return ideal_eq


class ParentAgraphIndv:
    def __init__(self, mapping_indv, P_desired):
        self.mapping_indv = mapping_indv
        self.P_desired = P_desired

    def evaluate_equation_at(self, x, detach=True):
        principal_stresses, state_parameters = x

        # mapping_matrices = torch.linalg.inv(self.mapping_indv.evaluate_equation_at_no_detach([state_parameters]))
        mapping_matrices = self.mapping_indv.evaluate_equation_at_no_detach([state_parameters])
        # P_mapped = torch.matmul(torch.matmul(torch.transpose(mapping_matrices, 1, 2), self.P_desired), mapping_matrices)
        P_mapped = torch.transpose(mapping_matrices, 1, 2) @ self.P_desired @ mapping_matrices
        yield_stresses = torch.transpose(principal_stresses, 1, 2) @ P_mapped @ principal_stresses
        # yield_stresses = torch.matmul(torch.matmul(torch.transpose(principal_stresses, 1, 2), P_mapped), principal_stresses)

        # try:
        #     # aniso -> vm
        #     # mapping_matrices = torch.linalg.inv(mapping_matrices)
        #
        #     # without inv, mapping matrices go from vm -> aniso
        #     mapped_stresses = mapping_matrices @ principal_stresses
        #
        #     yield_stresses = torch.transpose(mapped_stresses, 1, 2) @ self.P_desired @ mapped_stresses
        # except RuntimeError:
        #     yield_stresses = torch.full((principal_stresses.size(0), 1, 1), torch.inf)

        if detach:
            return yield_stresses.detach().numpy()
        else:
            return yield_stresses

    def evaluate_equation_with_x_gradient_at(self, x):
        for input in x:
            input.grad = None
            input.requires_grad = True

        yield_stresses = self.evaluate_equation_at(x, detach=False)

        if yield_stresses.requires_grad:
            yield_stresses.sum().backward()

        full_derivative = []
        for input in x:
            input_derivative = input.grad
            if input_derivative is None:
                try:
                    input_derivative = torch.zeros((yield_stresses.shape[0], input.shape[1]))
                except IndexError:
                    input_derivative = torch.zeros((yield_stresses.shape[0]))
            full_derivative.append(input_derivative.detach().numpy())

        for input in x:
            input.requires_grad = False

        return yield_stresses.detach().numpy(), full_derivative


class DirectMappingFitness(VectorBasedFunction):
    def __init__(self, *, explicit_fitness, implicit_fitness, P_desired):
        super().__init__(metric="rmse")
        self.implicit_fitness = implicit_fitness
        self.explicit_fitness = explicit_fitness
        # self.P_desired = P_desired[None, :, :].expand(len(implicit_fitness.training_data.x[0]), -1, -1)
        self.P_desired = P_desired

    def evaluate_fitness_vector(self, individual):
        parent_indv = ParentAgraphIndv(individual, self.P_desired)

        implicit = self.implicit_fitness.evaluate_fitness_vector(parent_indv)
        # implicit = self.implicit_fitness(parent_indv)

        # mean_output = np.mean(parent_indv.evaluate_equation_at(self.explicit_fitness.training_data.x).flatten())
        # self.explicit_fitness.training_data._y = mean_output
        explicit = self.explicit_fitness.evaluate_fitness_vector(parent_indv)
        # explicit = self.explicit_fitness(parent_indv)

        total_fitness = np.hstack((implicit, explicit))
        # total_fitness = implicit + explicit

        return total_fitness


class DoubleFitness(VectorBasedFunction):
    def __init__(self, *, explicit_fitness, implicit_fitness):
        super().__init__(metric="rmse")
        self.implicit_fitness = implicit_fitness
        self.explicit_fitness = explicit_fitness

    def evaluate_fitness_vector(self, individual):
        implicit = self.implicit_fitness.evaluate_fitness_vector(individual)
        explicit = self.explicit_fitness.evaluate_fitness_vector(individual)

        total_fitness = np.hstack((implicit, explicit))

        return total_fitness


from bingo.evaluation.fitness_function import VectorBasedFunction
class ParentAGraphFitness(VectorBasedFunction):
    def __init__(self, *, fitness_for_parent, P_desired):
        super().__init__(metric="mse")
        self.fitness_fn = fitness_for_parent
        self.P_desired = P_desired

    # def evaluate_fitness_vector(self, individual):
    #     parent_agraph = ParentAgraphIndv(individual, self.P_desired)
    #     fitness = self.fitness_fn.evaluate_fitness_vector(parent_agraph)
    #     return fitness

    def evaluate_fitness_vector(self, individual):
        parent_agraph = ParentAgraphIndv(individual, self.P_desired)
        mapping_matrices = individual.evaluate_equation_at([self.fitness_fn.training_data.x[1]])
        P = mapping_matrices.transpose((0, 2, 1)) @ self.P_desired.detach().numpy() @ mapping_matrices
        # normalization_fitness = P + P.transpose((0, 2, 1)) - 2 * np.mean(P, axis=(1, 2))[:, None, None]
        normalization_fitness = 2 * np.mean(P, axis=(1, 2)) / np.std(P + P.transpose((0, 2, 1)), axis=(1, 2))

        fitness = self.fitness_fn.evaluate_fitness_vector(parent_agraph)
        # TODO replace w/ std dev measure / mean?
        # if np.count_nonzero((np.abs(normalization_fitness) < 1e-4).flatten()) > 0.9 * normalization_fitness.size:
        #     fitness = np.full_like(fitness, np.inf)
        # return fitness
        return np.hstack((fitness, normalization_fitness))


def main():
    # dataset_path = "vpsc_evo_17_data_3d_points_transpose_implicit_format.txt"
    # dataset_path = "vpsc_evo_17_data_3d_points_implicit_format.txt"
    # dataset_path = "vpsc_evo_17_data_3d_points_implicit_format_shifted_idx.txt"
    # dataset_path = "vpsc_evo_0_data_3d_points_transpose_implicit_format.txt"
    # dataset_path = "vpsc_evo_0_data_3d_points_implicit_format.txt"
    dataset_path = "vpsc_evo_57_data_3d_points_implicit_format.txt"
    # dataset_path = "vpsc_evo_57_data_3d_points_transpose_implicit_format.txt"
    # dataset_path = "hill_w_hardening_transpose.txt"
    # dataset_path = "vm_w_hardening_transpose.txt"
    # dataset_path = "hill_w_hardening.txt"
    # dataset_path = "vpsc_evo_0_data_3d_points_implicit_format_shifted_idx.txt"
    data = np.loadtxt(dataset_path)
    print("running w/ dataset:", dataset_path)
    # data = data[~np.all(np.isnan(data), axis=1)]

    state_param_dims = [(0, 0)]

    output_dim = (3, 3)
    print("Dimensions of X variables:", state_param_dims)
    print("Dimension of output:", output_dim)

    x, dx_dt, _ = _calculate_partials(data, window_size=5)

    from VPSCWithHardeningParentAgraphImplicitExample import get_dx_dt
    # x = data[np.invert(np.all(np.isnan(data), axis=1))]
    # dx_dt = get_dx_dt(data)

    # data_2 = np.loadtxt("vpsc_evo_57_data_3d_points_implicit_format.txt")
    data_2 = np.loadtxt("vpsc_evo_57_data_3d_points_transpose_implicit_format.txt")
    # data_2 = np.loadtxt("hill_w_hardening.txt")
    # x_2 = data_2[np.invert(np.all(np.isnan(data_2), axis=1))]
    # dx_dt_2 = get_dx_dt(data_2)
    x_2, dx_dt_2, _ = _calculate_partials(data_2, window_size=5)

    x = np.vstack((x, x_2))
    dx_dt = np.vstack((dx_dt, dx_dt_2))

    implicit_training_data = ImplicitTrainingDataMD(x, dx_dt)
    x_0 = torch.from_numpy(implicit_training_data._x[:, :3].reshape((-1, 3, 1))).double()
    x_1 = torch.from_numpy(implicit_training_data._x[:, 3].reshape((-1))).double()
    x = [x_0, x_1]
    implicit_training_data._x = x
    implicit_fitness = ImplicitRegressionMD(implicit_training_data, required_params=4)

    y = np.ones((x_0.size(0), 1, 1))
    explicit_training_data = ExplicitTrainingDataMD(x, y)
    explicit_fitness = ExplicitRegressionMD(explicit_training_data)

    P_vm = np.array([[1, -0.5, -0.5],
                     [-0.5, 1, -0.5],
                     # [-0.5, -0.5, 1]])[None, :, :] / (100 * np.square(x_1.detach().numpy()) + 1)[:, None, None]
                     # [-0.5, -0.5, 1]])[None, :, :] / (100 * np.square(99 * x_1.detach().numpy() + 1))[:, None, None]
                     [-0.5, -0.5, 1]])
                     # [-0.5, -0.5, 1]])[None, :, :] / (100 + 99 * x_1.detach().numpy() + 1)[:, None, None]
                     # [-0.5, -0.5, 1]])[None, :, :] / (100 + np.square(x_1.detach().numpy()) + 1)[:, None, None]
    P_vm = torch.from_numpy(P_vm).double()


    # yield_surface_fitness = DirectMappingFitness(implicit_fitness=implicit_fitness,
    #                                              explicit_fitness=explicit_fitness,
    #                                              P_desired=P_vm)

    parent_explicit = ParentAGraphFitness(fitness_for_parent=explicit_fitness, P_desired=P_vm)
    parent_implicit = ParentAGraphFitness(fitness_for_parent=implicit_fitness, P_desired=P_vm)

    yield_surface_fitness = DoubleFitness(implicit_fitness=parent_implicit,
                                          explicit_fitness=parent_explicit)

    # from VPSCWithHardeningParentAgraphImplicitExample import ExplicitOptimizedImplicitFitness
    # local_opt_fitness = ContinuousLocalOptimizationMD(yield_surface_fitness, algorithm='lm', param_init_bounds=[-1, 1])
    # local_opt_fitness = ExplicitOptimizedImplicitFitness(parent_implicit, parent_explicit, algorithm="lm", param_init_bounds=[-1, 1])
    # local_opt_fitness = ExplicitOptimizedImplicitFitness(yield_surface_fitness, parent_explicit, algorithm="lm", param_init_bounds=[-1, 1])
    local_opt_fitness = ContinuousLocalOptimizationMD(parent_implicit, algorithm="lm", param_init_bounds=[-1, 1])
    # local_opt_fitness = ContinuousLocalOptimizationMD(yield_surface_fitness, algorithm="lm", param_init_bounds=[-1, 1])
    # local_opt_fitness = ContinuousLocalOptimizationMD(parent_explicit, algorithm="lm", param_init_bounds=[-1, 1])
    # local_opt_fitness = ContinuousLocalOptimizationMD(yield_surface_fitness, algorithm="lm", param_init_bounds=[-1, 1])
    evaluator = Evaluation(local_opt_fitness, multiprocess=5)

    ideal_eq = get_ideal_eq()
    # print("fitness before optimize:", yield_surface_fitness(ideal_eq))
    print("fitness before optimize:", parent_implicit(ideal_eq))
    print("fitness after optimize:", local_opt_fitness(ideal_eq))
    print(ideal_eq.constants)

    component_generator = ComponentGeneratorMD(state_param_dims, possible_dims=[(3, 3), (0, 0)])
    component_generator.add_operator("+")
    component_generator.add_operator("*")
    component_generator.add_operator("/")
    # component_generator.add_operator("sin")
    # component_generator.add_operator("cos")

    crossover = AGraphCrossoverMD()
    mutation = AGraphMutationMD(component_generator)
    agraph_generator = AGraphGeneratorMD(STACK_SIZE, component_generator, state_param_dims, output_dim,
                                         use_simplification=False, use_pytorch=True)

    ea = AgeFitnessEA(evaluator, agraph_generator, crossover, mutation, 0.3, 0.6, POP_SIZE)

    def agraph_similarity(ag_1, ag_2):
        """a similarity metric between agraphs"""
        return ag_1.fitness == ag_2.fitness and ag_1.get_complexity() == ag_2.get_complexity()

    pareto_front = ParetoFront(secondary_key=lambda ag: ag.get_complexity(),
                               similarity_function=agraph_similarity)

    island = Island(ea, agraph_generator, POP_SIZE, hall_of_fame=pareto_front)
    island.evolve(1)

    opt_result = island.evolve_until_convergence(max_generations=300,
                                                 fitness_threshold=1e-5)

    print(opt_result)
    best_indiv = island.get_best_individual()
    print(best_indiv.get_formatted_string("console"))
    print(best_indiv.fitness)
    print(repr(best_indiv.command_array))
    print(repr(best_indiv.constants))
    print(pareto_front)


if __name__ == '__main__':
    # import random
    # random.seed(7)
    # np.random.seed(7)

    main()
