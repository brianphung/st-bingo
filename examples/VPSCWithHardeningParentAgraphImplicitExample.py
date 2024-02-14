import dill

import numpy as np
import torch
torch.set_num_threads(1)
MULITPROCESS_CORES = 10

from bingo.evaluation.fitness_function import VectorBasedFunction
from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.serial_archipelago import SerialArchipelago
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island

from bingo.local_optimizers.continuous_local_opt_md import ContinuousLocalOptimizationMD
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.symbolic_regression.agraphMD.component_generator_md import ComponentGeneratorMD
from bingo.symbolic_regression.agraphMD.crossover_md import AGraphCrossoverMD
from bingo.symbolic_regression.agraphMD.generator_md import AGraphGeneratorMD
from bingo.symbolic_regression.agraphMD.mutation_md import AGraphMutationMD
from bingo.symbolic_regression.explicit_regression_md import ExplicitRegressionMD, ExplicitTrainingDataMD
from bingo.symbolic_regression.implicit_regression_md import ImplicitRegressionMD, ImplicitTrainingDataMD, _calculate_partials
# from bingo.symbolic_regression.implicit_regression_md import ImplicitRegressionMD, ImplicitTrainingDataMD
# from bingo.symbolic_regression.implicit_regression import _calculate_partials
from bingo.symbolic_regression.implicit_regression_schmidt import ImplicitRegressionSchmidt

from bingo.symbolic_regression.agraphMD.validation_backend import validation_backend

POP_SIZE = 250
STACK_SIZE = 15
# STACK_SIZE = 30


def get_ideal_eq():
    def _ideal_child_eq():
        cmd_arr = np.array([[1, 0, 3, 3],
                            [7, 0, 0, 0],
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
    from VPSCWithHardeningDirectExample import ParentAgraphIndv as MappingParent
    P_vm = np.array([[1, -0.5, -0.5],
                     [-0.5, 1, -0.5],
                     [-0.5, -0.5, 1]]) / 100
    P_vm = torch.from_numpy(P_vm).double()
    return MappingParent(_ideal_child_eq(), P_vm)


def get_ideal_eq():
    cmd_arr = np.array([[ 1,  0,  3,  3],
       [ 0,  0,  0,  0],
       [ 2,  1,  1,  1],
       [ 1,  1,  0,  0],
       [ 2,  2,  1,  0],
       [ 7,  3,  3,  1],
       [ 7,  0,  4,  0],
       [ 1,  2,  3,  3],
       [ 2,  3,  1,  0],
       [ 1,  3,  3,  3],
       [ 6,  7,  1,  0],
       [ 4,  4,  7,  0],
       [ 1,  4,  0,  0],
       [ 0,  0,  0,  0],
       [ 4,  2,  4,  1],
       [ 4,  0, 14,  1],
       [ 1,  5,  3,  3],
       [ 2, 15, 16,  1],
       [ 0,  0,  0,  0],
       [ 4, 17, 18,  1],
       [ 2, 19, 19,  0],
       [ 6, 20, 20,  0],
       [ 6, 21, 20,  0],
       [ 7, 22, 22,  0],
       [ 4, 23, 19,  0],
       [ 7, 11, 10,  0],
       [ 2, 21, 25,  0],
       [ 7, 26, 22,  0],
       [ 1,  6,  3,  3],
       [ 4, 27, 28,  1]])

    from bingo.symbolic_regression.agraphMD.pytorch_agraph_md import PytorchAGraphMD
    ideal_eq = PytorchAGraphMD(input_dims=[(0, 0)], output_dim=(3, 3))
    ideal_eq.command_array = cmd_arr
    ideal_eq._update()

    from numpy import array

    params = (array([[ 1277.92107988,  1977.87557786,  1251.54495874],
                     [-4420.60809802,  5053.07150027,   291.86703313],
                     [ 2094.16931078, -1531.84638842,  -133.52011265]]), array([[ -818.25593257,  1336.33875739,  -436.0463802 ],
                                                                                [ -248.25651909,   251.32870221, -1634.8903301 ],
                                                                                [-1223.91085177, -1059.63473912,   267.54630488]]), array([[ 71.72569292,  51.99625943,  45.47310858],
                                                                                                                                           [ 40.31926552, 149.81225251,  44.29981737],
                                                                                                                                           [ 66.03417915, -30.15735461,   7.02286306]]), array([[ -6.83027521,  -6.83046842,  -6.83747199],
                                                                                                                                                                                                [  4.23178559,   4.23190147,   4.23624162],
                                                                                                                                                                                                [-15.7528223 , -15.75325692, -15.76940778]]))

    # params = (np.array([[-584.14564259, 212.77774096, -248.99371655],
    #                  [522.07628579, 142.83283899, 270.16844954],
    #                  [181.03216541, -278.68249145, 532.69326429]]),
    #           np.array([[10.12171215, -7.4950326, -4.97661102],
    #                  [-16.19803523, -12.63443041, -12.44589146],
    #                  [-19.3387953, -8.43351292, 12.3503562]]),
    #           np.array([[3.82003457, -4.39652304, -5.26143143],
    #                  [-15.14788261, -2.69195116, -6.4691187],
    #                  [-18.93207797, -11.50411917, -14.59629573]]),
    #           np.array([[0.00033692, 0.00033692, 0.00033692],
    #                  [0.00203328, 0.00203328, 0.00203328],
    #                  [-0.0022005, -0.0022005, -0.0022005]]))
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

    params = (np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]), )
    flattened_params = np.hstack([param.flatten() for param in params])
    ideal_eq.set_local_optimization_params(flattened_params, [(3, 3)] * len(params))
    print(validation_backend.validate_individual(ideal_eq))
    print(ideal_eq)
    return ideal_eq
def get_ideal_eq():
    cmd_arr = np.array([[1, 0, 3, 3],
                        [4, 0, 0, 0],
                        [0, 0, 0, 0],
                        [6, 2, 2, 1],
                        [4, 1, 2, 1]])
    from bingo.symbolic_regression.agraphMD.pytorch_agraph_md import PytorchAGraphMD
    ideal_eq = PytorchAGraphMD(input_dims=[(0, 0)], output_dim=(3, 3))
    ideal_eq.command_array = cmd_arr
    ideal_eq._update()

    params = (np.array([[ 9.61045324e+34, -9.56764136e+34,  9.44130883e+34],
        [-1.03689872e+35, -6.49739827e+34,  4.86377030e+34],
        [-1.23112268e+35, -1.44892149e+35, -8.70809728e+34]]),)
    flattened_params = np.hstack([param.flatten() for param in params])
    ideal_eq.set_local_optimization_params(flattened_params, [(3, 3)] * len(params))
    print(validation_backend.validate_individual(ideal_eq))
    print(ideal_eq)
    return ideal_eq


class ParentAgraphIndv:
    def __init__(self, P_matrix_indv):
        self.P_matrix_indv = P_matrix_indv

    def evaluate_equation_at(self, x, detach=True):
        principal_stresses, state_parameters = x

        # P_vm = torch.from_numpy(np.array([[1, -0.5, -0.5],
        #                                   [-0.5, 1, -0.5],
        #                                   [-0.5, -0.5, 1]])[None, :, :])
        # P_matrices = self.P_matrix_indv.evaluate_equation_at_no_detach([state_parameters])
        # P_matrices = P_vm * P_matrices[:, None, None]

        try:
            P_matrices = self.P_matrix_indv.evaluate_equation_at_no_detach([state_parameters])

            yield_stresses = torch.transpose(principal_stresses, 1, 2) @ P_matrices @ principal_stresses
        except AttributeError:
            yield_stresses = self.P_matrix_indv.evaluate_equation_at(x, detach=detach)

        if detach:
            yield_stresses = yield_stresses.detach().numpy()
            # return yield_stresses / np.mean(yield_stresses)
            return yield_stresses
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


class YieldSurfaceParentAGraphFitness(VectorBasedFunction):
    def __init__(self, *, fitness_for_parent):
        super().__init__(metric="mse")
        self.fitness_fn = fitness_for_parent

    def evaluate_fitness_vector(self, individual):
        parent_agraph = ParentAgraphIndv(individual)
        P = individual.evaluate_equation_at([self.fitness_fn.training_data.x[1]])
        # normalization_fitness = 1 / (P + torch.transpose(P, 1, 2) - 2 * torch.mean(P, dim=0))
        # normalization_fitness = P + P.transpose((0, 2, 1)) - 2 * np.mean(P, axis=(1, 2))[:, None, None]

        # this is basically scale-corrected coefficient of variation
        normalization_fitness = 2 * np.mean(P, axis=(1, 2)) / np.std(P + P.transpose((0, 2, 1)), axis=(1, 2))

        # TODO why is this worse on constant matrices than the above?
        # normalization_fitness = (P + P.transpose((0, 2, 1))) / 2 * np.mean(P, axis=(1, 2))[:, None, None]
        # normalization_fitness = 1 / np.std(normalization_fitness, axis=(1, 2)).flatten()

        fitness = self.fitness_fn.evaluate_fitness_vector(parent_agraph)
        # if np.count_nonzero((np.abs(normalization_fitness) < 1e-5).flatten()) > 0.9 * normalization_fitness.size:
        #     fitness = np.full_like(fitness, np.inf)
        # return fitness
        return np.hstack((fitness, normalization_fitness))


class TotalYieldSurfaceParentAGraphFitness(VectorBasedFunction):
    def __init__(self, *, implicit_fitness, explicit_fitness):
        super().__init__(metric="rmse")
        self.implicit_fitness = implicit_fitness
        self.explicit_fitness = explicit_fitness

    def evaluate_fitness_vector(self, individual):
        parent_agraph = ParentAgraphIndv(individual)
        implicit_fitness = self.implicit_fitness.evaluate_fitness_vector(parent_agraph)
        # TODO motivation behind this line, was swapped for the above
        # implicit_fitness = self.implicit_fitness.evaluate_fitness_vector(individual)
        explicit_fitness = self.explicit_fitness.evaluate_fitness_vector(parent_agraph)
        return np.hstack((implicit_fitness, explicit_fitness))


class ExplicitOptimizedImplicitFitness(VectorBasedFunction):
    def __init__(self, implicit_fitness_function, explicit_fitness_function, algorithm='Nelder-Mead',
                 param_init_bounds=None, **optimization_options_kwargs):
        super().__init__(metric="mse")
        self._optimizer = ContinuousLocalOptimizationMD(explicit_fitness_function, algorithm,
                                                        param_init_bounds, **optimization_options_kwargs)
        self.end_fitness_function = implicit_fitness_function

    def evaluate_fitness_vector(self, individual):
        # mean_value = np.mean(self._optimizer._fitness_function.evaluate_fitness_vector(
        #     individual) + self._optimizer._fitness_function.fitness_fn.training_data.y)
        # self._optimizer._fitness_function.fitness_fn.training_data._y = np.repeat(mean_value[None, None, None], len(self._optimizer._fitness_function.fitness_fn.training_data.y), axis=0)
        optimizer_fitness = self._optimizer(individual)
        fitness = self.end_fitness_function.evaluate_fitness_vector(individual)
        return fitness


class MyYieldFitness(VectorBasedFunction):
    def __init__(self, *, data):
        super().__init__(metric="rmse")
        self.data = data

    def evaluate_fitness_vector(self, individual):
        principal_stresses, state_parameters = self.data
        P_matrices = individual.evaluate_equation_at([state_parameters])
        P_matrices = P_matrices / np.linalg.norm(P_matrices, axis=(1, 2))[:, None, None]

        yield_stresses = torch.transpose(principal_stresses, 1, 2) @ P_matrices @ principal_stresses
        yield_stresses = yield_stresses.detach().numpy().flatten()
        err = yield_stresses
        # err = yield_stresses / np.linalg.norm(P_matrices, axis=(1, 2))
        # err = yield_stresses - 1
        err = err.flatten()
        if np.any(np.isnan(err)):
            return np.full_like(err, np.inf)
        return err


def get_dx_dt(x_with_nan):
    point_trajectories = []

    point_trajectory = []
    for row in x_with_nan:
        if not np.all(np.isnan(row)):
            point_trajectory.append(row)
        else:
            point_trajectories.append(point_trajectory)
            point_trajectory = []
    point_trajectories.append(point_trajectory)  # for last point which doesn't have a nan footer

    dx_dts = []

    for point_trajectory in point_trajectories:
        dx_dt = []
        prev_point = point_trajectory[0]
        # prev_point = point_trajectory[0]
        for point in point_trajectory:
            dx_dt.append(point - prev_point)
            prev_point = point
        dx_dt.append(point_trajectory[-1] - point_trajectory[-2])
        dx_dt = dx_dt[1:]
        dx_dts.append(dx_dt)

    dx_dts = np.vstack(dx_dts)
    return dx_dts


def main():
    # TODO can have issues w/ optimization jacobian since combined fitness functions use two vectors?
    # normal implicit will find side view extension, transpose will find
    # trivial solution, need twisting to get both shape and extension properly
    # dataset_path = "vpsc_evo_17_data_3d_points_implicit_format.txt"
    # dataset_path = "vpsc_evo_17_data_3d_points_implicit_format_shifted_idx.txt"
    # dataset_path = "vpsc_evo_17_data_3d_points_transpose_implicit_format.txt"
    # dataset_path = "vpsc_evo_0_data_3d_points_transpose_implicit_format.txt"
    # dataset_path = "vpsc_evo_0_data_3d_points_transpose_implicit_format.txt"
    dataset_path = "vpsc_evo_57_data_3d_points_transpose_implicit_format.txt"
    # dataset_path = "vpsc_evo_0_data_3d_points_implicit_format.txt"
    # dataset_path = "vpsc_evo_0_data_3d_points_implicit_format_shifted_idx.txt"
    # dataset_path = "vpsc_evo_17_3d_points_w_extra_3d_points_transpose_implicit_format.txt"
    data = np.loadtxt(dataset_path)
    print("running w/ dataset:", dataset_path)

    # x, dx_dt, _ = _calculate_partials(data, window_size=5)
    # x, dx_dt, _ = _calculate_partials(data)
    # print(dx_dt.shape)

    x = data[np.invert(np.all(np.isnan(data), axis=1))]
    dx_dt = get_dx_dt(data)
    # up_vec = np.array([1, 1, 1])
    # dx_dt[:, :3] = np.array([np.cross(up_vec, vec) for vec in dx_dt[:, :3]])
    # first_yield_surface_i = np.arange(0, len(x), 9)

    # first_yield_surface_i = np.arange(0, 37, 1)
    # x = np.delete(x, first_yield_surface_i, axis=0)
    # dx_dt = np.delete(dx_dt, first_yield_surface_i, axis=0)

    data_2 = np.loadtxt("vpsc_evo_57_data_3d_points_implicit_format.txt")
    x_2 = data_2[np.invert(np.all(np.isnan(data_2), axis=1))]
    dx_dt_2 = get_dx_dt(data_2)
    # x_2, dx_dt_2, _ = _calculate_partials(data_2, window_size=5)

    x = np.vstack((x, x_2))
    dx_dt = np.vstack((dx_dt, dx_dt_2))

    implicit_training_data = ImplicitTrainingDataMD(x, dx_dt)
    x_0 = torch.from_numpy(implicit_training_data._x[:, :3].reshape((-1, 3, 1))).double()
    x_1 = torch.from_numpy(implicit_training_data._x[:, 3].reshape((-1))).double()
    x = [x_0, x_1]
    implicit_training_data._x = x
    # implicit_fitness = ImplicitRegressionMD(implicit_training_data, required_params=4)
    implicit_fitness = ImplicitRegressionMD(implicit_training_data)

    y = np.ones((x[0].shape[0], 1, 1)) * 100
    explicit_training_data = ExplicitTrainingDataMD(x, y)
    explicit_fitness = ExplicitRegressionMD(explicit_training_data, relative=True)

    parent_explicit_fitness = YieldSurfaceParentAGraphFitness(fitness_for_parent=explicit_fitness)
    parent_implicit_fitness = YieldSurfaceParentAGraphFitness(fitness_for_parent=implicit_fitness)
    # yield_surface_fitness = TotalYieldSurfaceParentAGraphFitness(implicit_fitness=implicit_fitness, explicit_fitness=explicit_fitness)
    yield_surface_fitness = TotalYieldSurfaceParentAGraphFitness(implicit_fitness=parent_implicit_fitness, explicit_fitness=explicit_fitness)
    my_fitness = MyYieldFitness(data=x)

    # ! yield surface fitness generally yields better results than parent implicit fitness
    # local_opt_fitness = ExplicitOptimizedImplicitFitness(yield_surface_fitness, parent_explicit_fitness, algorithm="lm")
    # local_opt_fitness = ExplicitOptimizedImplicitFitness(parent_implicit_fitness, parent_explicit_fitness, algorithm="lm")

    # local_opt_fitness = ContinuousLocalOptimizationMD(parent_explicit_fitness, algorithm='lm', param_init_bounds=[-1, 1])
    local_opt_fitness = ContinuousLocalOptimizationMD(parent_implicit_fitness, algorithm='lm', param_init_bounds=[-1, 1])
    # local_opt_fitness = ContinuousLocalOptimizationMD(my_fitness, algorithm='lm', param_init_bounds=[1, 1])
    # local_opt_fitness = ContinuousLocalOptimizationMD(yield_surface_fitness, algorithm='lm', param_init_bounds=[-1, 1])

    evaluator = Evaluation(local_opt_fitness, multiprocess=3)

    ideal_eq = get_ideal_eq()
    print("fitness before optimize:", parent_implicit_fitness(ideal_eq))
    # print("fitness before optimize:", yield_surface_fitness(ideal_eq))
    # print("\t explicit:", parent_explicit_fitness(ideal_eq))
    # print("\t implicit:", parent_implicit_fitness(ideal_eq))
    print("fitness after optimize:", local_opt_fitness(ideal_eq))
    # print("\t explicit:", parent_explicit_fitness(ideal_eq))
    # print("\t implicit:", parent_implicit_fitness(ideal_eq))
    print(repr(ideal_eq.constants))

    state_param_dims = [(0, 0)]
    output_dim = (3, 3)
    print("Dimensions of X variables:", state_param_dims)
    print("Dimension of output:", output_dim)

    component_generator = ComponentGeneratorMD(state_param_dims, possible_dims=[(3, 3), (0, 0)], possible_dim_weights=(0.4, 0.6))
    component_generator.add_operator("+")
    component_generator.add_operator("*")
    # component_generator.add_operator("/")
    component_generator.add_operator("sin")
    component_generator.add_operator("cos")

    crossover = AGraphCrossoverMD()
    mutation = AGraphMutationMD(component_generator)
    agraph_generator = AGraphGeneratorMD(STACK_SIZE, component_generator, state_param_dims, output_dim,
                                         use_simplification=False, use_pytorch=True)

    ea = AgeFitnessEA(evaluator, agraph_generator, crossover, mutation, 0.3, 0.6, POP_SIZE)

    island = Island(ea, agraph_generator, POP_SIZE)
    island.evolve(1)

    opt_result = island.evolve_until_convergence(max_generations=10000,
                                                 fitness_threshold=1e-5)

    print(opt_result)
    best_indiv = island.get_best_individual()
    print(best_indiv.get_formatted_string("console"))
    print(best_indiv.fitness)
    print(repr(best_indiv.command_array))
    print(repr(best_indiv.constants))


if __name__ == '__main__':
    # import random
    # random.seed(7)
    # np.random.seed(7)

    main()
