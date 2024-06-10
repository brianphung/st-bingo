from math import floor
import os

import numpy as np
import torch
torch.set_num_threads(1)

import multiprocessing
CPU_COUNT = multiprocessing.cpu_count()

from bingo.evaluation.fitness_function import VectorBasedFunction
from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island
from bingo.stats.pareto_front import ParetoFront

from bingo.local_optimizers.continuous_local_opt_md import ContinuousLocalOptimizationMD
from bingo.symbolic_regression.agraphMD.component_generator_md import ComponentGeneratorMD
from bingo.symbolic_regression.agraphMD.crossover_md import AGraphCrossoverMD
from bingo.symbolic_regression.agraphMD.generator_md import AGraphGeneratorMD
from bingo.symbolic_regression.agraphMD.mutation_md import AGraphMutationMD
from bingo.symbolic_regression.explicit_regression_md import ExplicitRegressionMD, ExplicitTrainingDataMD
from bingo.symbolic_regression.implicit_regression_md import ImplicitRegressionMD, ImplicitTrainingDataMD, \
    _calculate_partials

POP_SIZE = 100
STACK_SIZE = 10


class ParentAgraphIndv:
    """
    Parent agraph that takes in a mapping individual that returns mapping
    matrices based on state parameters (plastic strain measure in our case). It
    uses the mapping matrices to transform P_desired/P_fict into P_real.

    Basically this takes mapping individual $f(\alpha)$ (where $\alpha$ are
    state parameters) and converts it to $f(\alpha).T @ P_desired @ f(\alpha)$
    """
    def __init__(self, mapping_indv, P_desired):
        self.mapping_indv = mapping_indv
        self.P_desired = P_desired

    def evaluate_equation_at(self, x, detach=True):
        principal_stresses, state_parameters = x

        mapping_matrices = self.mapping_indv.evaluate_equation_at_no_detach([state_parameters])
        P_mapped = torch.transpose(mapping_matrices, 1, 2) @ self.P_desired @ mapping_matrices
        yield_stresses = torch.transpose(principal_stresses, 1, 2) @ P_mapped @ principal_stresses

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
    """
    Evaluates a given mapping individual by converting it to a parent agraph
    as described above and then evaluating said agraph on the provided
    explicit and implicit fitness functions.
    """
    def __init__(self, *, explicit_fitness, implicit_fitness, P_desired):
        super().__init__(metric="rmse")
        self.implicit_fitness = implicit_fitness
        self.explicit_fitness = explicit_fitness
        self.P_desired = P_desired

    def evaluate_fitness_vector(self, individual):
        parent_indv = ParentAgraphIndv(individual, self.P_desired)

        implicit = self.implicit_fitness.evaluate_fitness_vector(parent_indv)
        explicit = self.explicit_fitness.evaluate_fitness_vector(parent_indv)

        total_fitness = np.hstack((implicit, explicit))
        return total_fitness


class ParentFitnessToChildFitness(VectorBasedFunction):
    """
    Transforms a fitness function for the parent into a fitness
    function for its child by converting the child to a parent agraph
    before evaluating it on the provided fitness function for the parent.
    """
    def __init__(self, *, fitness_for_parent, P_desired):
        super().__init__(metric="mse")
        self.fitness_fn = fitness_for_parent
        self.P_desired = P_desired

    def evaluate_fitness_vector(self, individual):
        parent_agraph = ParentAgraphIndv(individual, self.P_desired)

        # evaluate the child individual on the state parameter to get the mapping matrices
        mapping_matrices = individual.evaluate_equation_at([self.fitness_fn.training_data.x[1]])

        P_mapped = mapping_matrices.transpose((0, 2, 1)) @ self.P_desired.detach().numpy() @ mapping_matrices

        # normalize against the plane solution by using inverse of coefficient of variation
        normalization_fitness = 2 * np.mean(P_mapped, axis=(1, 2)) / np.std(P_mapped + P_mapped.transpose((0, 2, 1)), axis=(1, 2))

        fitness = self.fitness_fn.evaluate_fitness_vector(parent_agraph)
        return np.hstack((fitness, normalization_fitness))


class DoubleFitness(VectorBasedFunction):
    """
    Evaluates the provided individual on both the provided
    explicit and implicit fitness functions and returns the concatenated fitness
    vectors of them
    """
    def __init__(self, *, explicit_fitness, implicit_fitness):
        super().__init__(metric="mse")
        self.implicit_fitness = implicit_fitness
        self.explicit_fitness = explicit_fitness

    def evaluate_fitness_vector(self, individual):
        implicit = self.implicit_fitness.evaluate_fitness_vector(individual)
        explicit = self.explicit_fitness.evaluate_fitness_vector(individual)

        total_fitness = np.hstack((implicit, explicit))

        return total_fitness


def run_experiment(dataset_path,
                   transposed_dataset_path,
                   max_generations=100,
                   checkpoint_path="checkpoints"):
    # load data
    data = np.loadtxt(dataset_path)
    transposed_data = np.loadtxt(transposed_dataset_path)
    print("running w/ dataset:", dataset_path)

    state_param_dims = [(1, 1)]
    output_dim = (3, 3)

    # get local derivatives from data
    x, dx_dt, _ = _calculate_partials(data, window_size=5)
    x_transposed, dx_dt_transposed, _ = _calculate_partials(transposed_data, window_size=5)

    # combine normal and transposed data
    x = np.vstack((x, x_transposed))
    dx_dt = np.vstack((dx_dt, dx_dt_transposed))

    # implicit fitness function to match local derivatives
    implicit_training_data = ImplicitTrainingDataMD(x, dx_dt)

    # convert numpy arrays into pytorch tensors
    x_0 = torch.from_numpy(implicit_training_data._x[:, :3].reshape((-1, 3, 1))).double()
    x_1 = torch.from_numpy(implicit_training_data._x[:, 3].reshape((-1))).double()
    x = [x_0, x_1]
    implicit_training_data._x = x
    implicit_fitness = ImplicitRegressionMD(implicit_training_data, required_params=4)

    # explicit fitness function to make yield stress constant per yield surface
    y = np.ones((x_0.size(0), 1, 1))
    explicit_training_data = ExplicitTrainingDataMD(x, y)
    explicit_fitness = ExplicitRegressionMD(explicit_training_data)

    P_vm = np.array([[1, -0.5, -0.5],
                     [-0.5, 1, -0.5],
                     [-0.5, -0.5, 1]])
    P_vm = torch.from_numpy(P_vm).double()

    # convert fitness functions to child fitness functions
    parent_explicit = ParentFitnessToChildFitness(fitness_for_parent=explicit_fitness, P_desired=P_vm)
    parent_implicit = ParentFitnessToChildFitness(fitness_for_parent=implicit_fitness, P_desired=P_vm)

    # combine implicit and explicit fitness functions
    yield_surface_fitness = DoubleFitness(implicit_fitness=parent_implicit, explicit_fitness=parent_explicit)

    local_opt_fitness = ContinuousLocalOptimizationMD(yield_surface_fitness, algorithm="lm", param_init_bounds=[-1, 1])

    # downscale CPU_COUNT to avoid resource conflicts
    N_CPUS_TO_USE = floor(CPU_COUNT * 0.95)
    print(f"using {N_CPUS_TO_USE}/{CPU_COUNT} cpus")
    evaluator = Evaluation(local_opt_fitness, multiprocess=N_CPUS_TO_USE)

    # setup archipelago
    component_generator = ComponentGeneratorMD(state_param_dims, possible_dims=[(3, 3), (1, 1)])
    component_generator.add_operator("+")
    component_generator.add_operator("*")
    component_generator.add_operator("/")

    crossover = AGraphCrossoverMD()
    mutation = AGraphMutationMD(component_generator)
    agraph_generator = AGraphGeneratorMD(STACK_SIZE, component_generator, state_param_dims, output_dim,
                                         use_simplification=False, use_pytorch=True, use_symmetric_constants=True)

    ea = AgeFitnessEA(evaluator, agraph_generator, crossover, mutation, 0.3, 0.6, POP_SIZE)

    def agraph_similarity(ag_1, ag_2):
        return ag_1.fitness == ag_2.fitness and ag_1.get_complexity() == ag_2.get_complexity()

    pareto_front = ParetoFront(secondary_key=lambda ag: ag.get_complexity(),
                               similarity_function=agraph_similarity)

    island = Island(ea, agraph_generator, POP_SIZE, hall_of_fame=pareto_front)

    # start run
    island.evolve(1)
    island.evolve_until_convergence(max_generations=max_generations,
                                    fitness_threshold=1e-5,
                                    convergence_check_frequency=10,
                                    num_checkpoints=3,
                                    checkpoint_base_name=f"{checkpoint_path}/checkpoint")

    print("Finished bingo run, pareto front is:")
    print(pareto_front)


if __name__ == '__main__':
    # make checkpoint folders if they don't exist
    # vpsc_checkpoint_path = "checkpoints/vpsc75"
    # if not os.path.exists(vpsc_checkpoint_path):
    #     os.makedirs(vpsc_checkpoint_path)

    hill_checkpoint_path = "checkpoints/hill"
    if not os.path.exists(hill_checkpoint_path):
        os.makedirs(hill_checkpoint_path)

    # # run vpsc experiment
    # vpsc_data_path = "../data/processed_data/vpsc_75_bingo_format.txt"
    # vpsc_transposed_data_path = "../data/processed_data/vpsc_75_transpose_bingo_format.txt"
    # run_experiment(vpsc_data_path,
    #                vpsc_transposed_data_path,
    #                max_generations=300,
    #                checkpoint_path=vpsc_checkpoint_path)

    # run hill experiment
    hill_data_path = "../data/processed_data/hill_w_hardening.txt"
    hill_transposed_data_path = "../data/processed_data/hill_w_hardening_transpose.txt"
    run_experiment(hill_data_path,
                   hill_transposed_data_path,
                   max_generations=100,
                   checkpoint_path=hill_checkpoint_path)
