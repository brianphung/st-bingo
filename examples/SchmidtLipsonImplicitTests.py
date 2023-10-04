import numpy as np

from VPSCWithHardeningParentAgraphImplicitExample import get_dx_dt

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
    AGraphMutation, \
    ImplicitRegression, \
    ImplicitTrainingData

POP_SIZE = 100
STACK_SIZE = 10


def generate_circle_data(n_points):
    t = np.linspace(0, 2 * np.pi, num=n_points)
    points = np.hstack((np.cos(t)[:, None], np.sin(t)[:, None]))
    return points

def get_ideal_eq():
    from bingo.symbolic_regression.agraph.agraph import AGraph
    ideal_eq = AGraph(equation="X_0 * X_0 + X_1 * X_1")
    return ideal_eq


if __name__ == "__main__":
    x = generate_circle_data(100)
    dx_dt = get_dx_dt(x)
    # dx_dt = 2 * x
    # dx_dt = x
    def get_rot_mat(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])

    # dx_dt = (get_rot_mat(-np.pi / 2.) @ dx_dt.T).T

    up_vec = np.array([0, 0, -1])
    dx_dt = np.array([np.cross(up_vec, np.hstack((dxdt_vec[None, :], np.zeros(1)[None, :])).flatten()) for dxdt_vec in dx_dt])[:, :2]
    import matplotlib.pyplot as plt
    plt.scatter(*x.T)
    plt.quiver(*x.T, *dx_dt.T, scale=1, scale_units="xy")
    plt.show()

    component_generator = ComponentGenerator(x.shape[1])
    component_generator.add_operator("+")
    component_generator.add_operator("-")
    component_generator.add_operator("*")

    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)

    agraph_generator = AGraphGenerator(STACK_SIZE, component_generator, use_simplification=True)

    training_data = ImplicitTrainingData(x, dx_dt)
    fitness = ImplicitRegression(training_data=training_data)
    optimizer = ScipyOptimizer(fitness, method="lm")
    local_opt_fitness = LocalOptFitnessFunction(fitness, optimizer)

    ideal_eq = get_ideal_eq()
    print("fitness of ideal eq:", local_opt_fitness(ideal_eq))
    evaluator = Evaluation(local_opt_fitness)

    ea = AgeFitnessEA(evaluator, agraph_generator, crossover,
                      mutation, 0.4, 0.4, POP_SIZE)

    island = Island(ea, agraph_generator, POP_SIZE)
    archipelago = SerialArchipelago(island)

    opt_result = archipelago.evolve_until_convergence(max_generations=500,
                                                      fitness_threshold=1.0e-4)
    print(opt_result)
    print(opt_result.ea_diagnostics)
    best_individual = archipelago.get_best_individual()
    print(f"found {best_individual} with fitness {best_individual.fitness}")
    print(f"\tcmd arr: {best_individual.command_array}\nconstants: {best_individual.constants}")
