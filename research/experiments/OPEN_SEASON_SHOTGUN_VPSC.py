from math import floor
import os

import numpy as np
import torch
torch.set_num_threads(1)
import sys
import scipy

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

POP_SIZE = 300
STACK_SIZE = 20
sqrt2 = np.sqrt(2)
sqrt3 = np.sqrt(3)
upper_triangle_idx  =np.triu_indices(5)

d_trans = 1.0/np.sqrt(6) * np.array( [[ sqrt2, sqrt2, sqrt2, 0, 0, 0  ],
                                [ -1, -1, 2, 0, 0, 0],
                                [-sqrt3, sqrt3, 0, 0, 0, 0],
                                [0, 0, 0, 2*sqrt3, 0, 0],
                                [0, 0, 0, 0, 2*sqrt3, 0],
                                [0, 0, 0, 0, 0, 2*sqrt3] ])

d_inv = 1.0/np.sqrt(6) * np.array( [[ sqrt2, -1, -sqrt3, 0, 0, 0  ],
                                   [ sqrt2, -1, sqrt3, 0, 0, 0],
                                   [ sqrt2, 2, 0, 0, 0, 0],
                                   [0, 0, 0, sqrt3, 0, 0],
                                   [0, 0, 0, 0, sqrt3, 0],
                                   [0, 0, 0, 0, 0, sqrt3] ])

class ParentAgraphIndv:
    """
    Parent agraph that takes in a mapping individual that returns mapping
    matrices based on state parameters (plastic strain measure in our case). It
    uses the mapping matrices to transform P_desired/P_fict into P_real.

    Basically this takes mapping individual $f(\alpha)$ (where $\alpha$ are
    state parameters) and converts it to $f(\alpha).T @ P_desired @ f(\alpha)$
    """
    def __init__(self, mapping_indv):
        self.mapping_indv = mapping_indv

    def evaluate_equation_at(self, x, detach=True):
        
        principal_stresses, state_parameters = x
        # print('pricipal stresses=',principal_stresses)
        P_mapped = self.mapping_indv.evaluate_equation_at_no_detach([state_parameters])
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


class ParentFitnessToChildFitness(VectorBasedFunction):
    """
    Transforms a fitness function for the parent into a fitness
    function for its child by converting the child to a parent agraph
    before evaluating it on the provided fitness function for the parent.
    """
    def __init__(self, *, fitness_for_parent):
        super().__init__(metric="mse")
        self.fitness_fn = fitness_for_parent

    def evaluate_fitness_vector(self, individual):
        parent_agraph = ParentAgraphIndv(individual)

        # Evaluate the fitness the explicit yield points
        fitness = self.fitness_fn.evaluate_fitness_vector(parent_agraph)

        eqps_max = np.amax(self.fitness_fn.training_data.x[1].detach().numpy())
        number_of_granular_points = len(self.fitness_fn.training_data.x[1]) # This isn't actually the number of eqps points, it's number of input points TOTAL 

        x_gran = np.linspace(0, eqps_max,  number_of_granular_points)

        # evaluate the child individual on the granular eqps to get the mapping matrices
        P_mapped_dev = individual.evaluate_equation_at(np.array([x_gran]))
        P_template = np.zeros((6,6))
        for P_candidate_dev in P_mapped_dev:
            P_template[1:, 1:] = P_candidate_dev
            P_candidate = d_trans.T @ P_template @ d_trans
            _, info = scipy.linalg.lapack.dpotrf(P_candidate + np.eye(P_candidate.shape[1]) * 1e-16)
            if info != 0:
                fitness *= np.inf
                return fitness

        return fitness #np.hstack((fitness, normalization_fitness))


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


def remove_hydrostatic_dependence(data, transpose_data):
    # Get the rotation matrix with full precision

    # Isolate stress
    stress_vectors = data[:,0:6]
    stress_vectors_T = transpose_data[:, 0:6]

    # # Shift the X_0 up a tiny bit
    # data[:,6] += 1e-6
    # transpose_data[:,6] += 1e-6

    # Rotate to natural basis
    stress_vectors = (d_trans @ stress_vectors.T).T
    stress_vectors_T = (d_trans @ stress_vectors_T.T).T
    #print(np.shape(stress_vectors_T), np.shape(transpose_data[:,3].T))

    # We assert that there should be hydrostatic stress independence. Therefore, we only solve for the 5x5 submatrix
    data = np.column_stack([stress_vectors[:,1:6], data[:,6].T ])
    transpose_data = np.column_stack( [stress_vectors_T[:, 1:6], transpose_data[:,6].T ])

    return data, transpose_data


def run_deviatoric_experiment(dataset_path,
                   transposed_dataset_path,
                   processors,
                   max_generations=100,
                   checkpoint_path="checkpoints"
                   ):
    
    # load data
    data_pre = np.loadtxt(dataset_path)
    transposed_data_pre = np.loadtxt(transposed_dataset_path)
    print("running w/ dataset:", dataset_path)

    data, transposed_data = remove_hydrostatic_dependence(data_pre, transposed_data_pre)
    #print(data, transposed_data)

    state_param_dims = [(1, 1)]
    output_dim = (5, 5)
    #print(transposed_data)

    # get local derivatives from data
    x, dx_dt, _ = _calculate_partials(data, window_size=5)
    #x_transposed, dx_dt_transposed, _ = _calculate_partials(transposed_data, window_size=5)


    # implicit fitness function to match local derivatives
    implicit_training_data = ImplicitTrainingDataMD(x, dx_dt)

    # convert numpy arrays into pytorch tensors
    x_0 = torch.from_numpy(implicit_training_data._x[:, :5].reshape((-1, 5, 1))).double()
    x_1 = torch.from_numpy(implicit_training_data._x[:, 5].reshape((-1,1,1))).double()
    x = [x_0, x_1]
    #print('x_0=',x_0[0])
    # #print('x_1=',x_1)
    implicit_training_data._x = x
    implicit_fitness = ImplicitRegressionMD(implicit_training_data) #, required_params=2)

    # explicit fitness function to make yield stress constant per yield surface
    y = np.ones((x_0.size(0), 1, 1))
    explicit_training_data = ExplicitTrainingDataMD(x, y)
    explicit_fitness = ExplicitRegressionMD(explicit_training_data)

    # P_vm_nat = 3./2.*np.eye(5)

    # P_vm = torch.from_numpy(P_vm_nat).double()



    # convert fitness functions to child fitness functions
    parent_explicit = ParentFitnessToChildFitness(fitness_for_parent=explicit_fitness)
    parent_implicit = ParentFitnessToChildFitness(fitness_for_parent=implicit_fitness)

    # combine implicit and explicit fitness functions
    yield_surface_fitness = DoubleFitness(implicit_fitness=parent_implicit, explicit_fitness=parent_explicit)

    local_opt_fitness = ContinuousLocalOptimizationMD(yield_surface_fitness, algorithm="lm", param_init_bounds=[-1, 1])
    #local_opt_fitness = ContinuousLocalOptimizationMD(yield_surface_fitness, algorithm="lm", param_init_bounds=[-1, 1], options={"ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12, "maxiter": 10000})

    # downscale CPU_COUNT to avoid resource conflicts
    N_CPUS_TO_USE = processors
    print(f"using {N_CPUS_TO_USE}/{CPU_COUNT} cpus")
    evaluator = Evaluation(local_opt_fitness, multiprocess=N_CPUS_TO_USE)

    # setup archipelago
    component_generator = ComponentGeneratorMD(state_param_dims, possible_dims=[(5, 5), (1, 1)])
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
                                    num_checkpoints=None,
                                    checkpoint_base_name=f"{checkpoint_path}/checkpoint")
    f=open('output.txt','w')
    print("Finished bingo run, pareto front is:")
    print(pareto_front)
    print(pareto_front,file=f)
    print('best individual=',island.get_best_individual())

    f.close()
    


if __name__ == '__main__':

    processors = int(sys.argv[1])
    shotgun_number = int(sys.argv[2])

    vpsc_checkpoint_path = f"checkpoints/vpsc_hcp_OPENSEASON_BIIIIIIG_Stack_{shotgun_number}"
    if not os.path.exists(vpsc_checkpoint_path):
        os.makedirs(vpsc_checkpoint_path)

    #run hill experiment
    data_path = "/uufs/chpc.utah.edu/common/home/u0674703/scv/bingo_shotgun_vpsc/st-bingo/research/data_6x6/processed_data/VPSC_HCP_BINGO_shift.txt"

    run_deviatoric_experiment(data_path,
                data_path,
                max_generations=5000,
                checkpoint_path=vpsc_checkpoint_path,
                processors=processors)
