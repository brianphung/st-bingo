import numpy as np
import torch
torch.set_num_threads(1)

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
STACK_SIZE = 15


class ParentAgraphIndv:
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


from bingo.evaluation.fitness_function import VectorBasedFunction
class ParentAGraphFitness(VectorBasedFunction):
    def __init__(self, *, fitness_for_parent, P_desired):
        super().__init__(metric="mse")
        self.fitness_fn = fitness_for_parent
        self.P_desired = P_desired

    def evaluate_fitness_vector(self, individual):
        parent_agraph = ParentAgraphIndv(individual, self.P_desired)
        mapping_matrices = individual.evaluate_equation_at([self.fitness_fn.training_data.x[1]])

        P_mapped = mapping_matrices.transpose((0, 2, 1)) @ self.P_desired.detach().numpy() @ mapping_matrices

        normalization_fitness = 2 * np.mean(P_mapped, axis=(1, 2)) / np.std(P_mapped + P_mapped.transpose((0, 2, 1)), axis=(1, 2))

        fitness = self.fitness_fn.evaluate_fitness_vector(parent_agraph)
        return np.hstack((fitness, normalization_fitness))


def main():
    dataset_path = "../data/vpsc_evo_57_data_3d_points_implicit_format.txt"
    data = np.loadtxt(dataset_path)
    print("running w/ dataset:", dataset_path)

    state_param_dims = [(0, 0)]

    output_dim = (3, 3)
    print("Dimensions of X variables:", state_param_dims)
    print("Dimension of output:", output_dim)

    x, dx_dt, _ = _calculate_partials(data, window_size=5)

    transposed_data = np.loadtxt("../data/vpsc_evo_57_data_3d_points_transpose_implicit_format.txt")
    x_transposed, dx_dt_transposed, _ = _calculate_partials(transposed_data, window_size=5)

    x = np.vstack((x, x_transposed))
    dx_dt = np.vstack((dx_dt, dx_dt_transposed))

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
                     [-0.5, -0.5, 1]])
    P_vm = torch.from_numpy(P_vm).double()


    yield_surface_fitness = DirectMappingFitness(implicit_fitness=implicit_fitness,
                                                 explicit_fitness=explicit_fitness,
                                                 P_desired=P_vm)

    local_opt_fitness = ContinuousLocalOptimizationMD(yield_surface_fitness, algorithm="lm", param_init_bounds=[-1, 1])
    evaluator = Evaluation(local_opt_fitness, multiprocess=5)

    component_generator = ComponentGeneratorMD(state_param_dims, possible_dims=[(3, 3), (0, 0)])
    component_generator.add_operator("+")
    component_generator.add_operator("*")
    component_generator.add_operator("/")

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

    island.evolve_until_convergence(max_generations=10, fitness_threshold=1e-5)

    print("Finished bingo run, pareto front is:")
    print(pareto_front)


if __name__ == '__main__':
    # import random
    # random.seed(7)
    # np.random.seed(7)

    main()
