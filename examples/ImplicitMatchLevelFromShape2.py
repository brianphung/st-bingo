import numpy as np

from bingo.evaluation.fitness_function import VectorBasedFunction
from bingo.symbolic_regression.implicit_regression import ImplicitRegression, ImplicitTrainingData
from bingo.symbolic_regression.agraph.agraph import AGraph

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
    AGraphMutation

POP_SIZE = 250
STACK_SIZE = 10


def get_ideal_eq():
    # TODO these shouldn't give different results!!!!
    # ideal_eq = AGraph(equation="X_0**2 + X_1**2")
    ideal_eq = AGraph(equation="X_0*X_0 + X_1*X_1")
    return ideal_eq


def get_shape_eq_outputs(circle_points):
    shape_eq = get_ideal_eq()
    outputs = shape_eq.evaluate_equation_at(circle_points)
    return outputs


def get_circle_data(implicit=True):
    t = np.linspace(0, 2 * np.pi, num=10)
    latent_variables = np.linspace(1, 10, num=10)
    circle_data = np.empty((latent_variables.shape[0], t.shape[0], 3))
    for i, latent_variable in enumerate(latent_variables):
        radius = latent_variable ** 2 + latent_variable
        x = radius * np.cos(t)
        y = radius * np.sin(t)
        extended_latent_variable = np.ones((t.shape[0], 1)) * latent_variable
        points_and_latent = np.hstack((x[:, None], y[:, None], extended_latent_variable))
        circle_data[i] = points_and_latent

    if implicit:
        # for non-transposed version (match expansion instead of shape)
        circle_data = circle_data.transpose((1, 0, 2))

        implicit_data = []
        for circle_point_set in circle_data:
            implicit_data.extend(circle_point_set)
            implicit_data.append(np.full(circle_data.shape[2], np.nan))
        implicit_data = np.array(implicit_data[:-1])
        return implicit_data
    else:
        circle_data = circle_data.reshape((-1, circle_data.shape[2]))
        return circle_data


class ParentFitness(VectorBasedFunction):
    def __init__(self, *, child_agraph, fitness_for_parent):
        super().__init__(None, metric="mae")
        self.child_agraph_str = "(" + child_agraph.get_formatted_string(format_="sympy") + ")"
        self.fitness_fn = fitness_for_parent

    def evaluate_fitness_vector(self, individual):
        parent_string = individual.get_formatted_string(format_="sympy")
        # full_agraph = AGraph(equation=self.child_agraph_str + " / (" + parent_string + ")")
        full_agraph = AGraph(equation=self.child_agraph_str + " - (" + parent_string + ")")
        return self.fitness_fn.evaluate_fitness_vector(full_agraph)


class CombinedFitness(VectorBasedFunction):
    def __init__(self, first_fitness_fn, second_fitness_fn):
        super().__init__(None, "mae")
        self.first_fitness = first_fitness_fn
        self.second_fitness = second_fitness_fn

    def evaluate_fitness_vector(self, individual):
        first_fitness_eval = self.first_fitness.evaluate_fitness_vector(individual)
        second_fitness_eval = self.second_fitness.evaluate_fitness_vector(individual)
        return np.hstack((first_fitness_eval, second_fitness_eval))


def main():
    use_implicit = True
    circle_data = get_circle_data(implicit=use_implicit)
    x = circle_data

    implicit_training_data = ImplicitTrainingData(x)
    normal_implicit_fitness = ImplicitRegression(implicit_training_data)
    implicit_parent_fitness = ParentFitness(child_agraph=get_ideal_eq(), fitness_for_parent=normal_implicit_fitness)

    fitness = implicit_parent_fitness

    eq = AGraph(equation="(X_2 * X_2 + X_2)**2")
    print(fitness(eq))

    optimizer = ScipyOptimizer(fitness, method="lm", param_init_bounds=[-1, 1])
    local_opt_fitness = LocalOptFitnessFunction(fitness, optimizer)
    evaluator = Evaluation(local_opt_fitness)

    component_generator = ComponentGenerator(3)
    component_generator.add_operator("+")
    component_generator.add_operator("-")
    component_generator.add_operator("*")
    component_generator.add_operator("/")

    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)

    agraph_generator = AGraphGenerator(STACK_SIZE, component_generator,
                                       use_simplification=False)

    ea = AgeFitnessEA(evaluator, agraph_generator, crossover,
                      mutation, 0.4, 0.4, POP_SIZE)

    island = Island(ea, agraph_generator, POP_SIZE)
    archipelago = SerialArchipelago(island)

    opt_result = archipelago.evolve_until_convergence(max_generations=500,
                                                      fitness_threshold=1.0e-4)
    print(opt_result)
    best_individual = archipelago.get_best_individual()
    sympy_str = best_individual.get_formatted_string("sympy")
    print("found individual:", sympy_str)
    import sympy
    print("simplified individual:", sympy.simplify(sympy_str))


if __name__ == '__main__':
    main()
