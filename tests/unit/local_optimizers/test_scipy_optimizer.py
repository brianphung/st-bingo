# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
from copy import deepcopy

import pytest
import numpy as np

from scipy import optimize
from scipy.optimize import OptimizeResult

from bingo.evaluation.fitness_function \
    import FitnessFunction, VectorBasedFunction
from bingo.evaluation.gradient_mixin import GradientMixin, VectorGradientMixin
from bingo.local_optimizers.continuous_local_opt import ChromosomeInterface
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer, \
    ROOT_SET, MINIMIZE_SET


class DummyLocalOptIndividual(ChromosomeInterface):
    def needs_local_optimization(self):
        return True

    def get_number_local_optimization_params(self):
        return 1

    def set_local_optimization_params(self, params):
        try:
            self.param = params[0]
        except IndexError:  # for issue with powell
            self.param = params


class BloatedOptIndividual(ChromosomeInterface):
    def __init__(self):
        self.param = [1, 2, 3]
        self._fitness = None
        self._fit_set = False

    def needs_local_optimization(self):
        return True

    def get_number_local_optimization_params(self):
        return 3

    def set_local_optimization_params(self, params):
        self.param = params


@pytest.mark.parametrize("obj_func_type, raises_error",
                         [(FitnessFunction, True),
                          (VectorBasedFunction, False)])
def test_valid_objective_function_init(mocker, obj_func_type, raises_error):
    mocked_obj_fn = mocker.create_autospec(obj_func_type)
    if raises_error:
        with pytest.raises(TypeError):
            _ = ScipyOptimizer(mocked_obj_fn, method="lm")
    else:
        _ = ScipyOptimizer(mocked_obj_fn, method="lm")


@pytest.mark.parametrize("obj_func_type, raises_error",
                         [(FitnessFunction, True),
                          (VectorBasedFunction, False)])
def test_valid_objective_function_property(mocker, obj_func_type, raises_error):
    mocked_obj_fn = mocker.create_autospec(obj_func_type)

    # construct obj with valid objective fn
    valid_obj_fn = mocker.create_autospec(VectorBasedFunction)
    opt = ScipyOptimizer(valid_obj_fn, method="lm")

    # test setting obj fn with property
    if raises_error:
        with pytest.raises(TypeError):
            opt.objective_fn = mocked_obj_fn
    else:
        opt.objective_fn = mocked_obj_fn


def test_invalid_method(mocker):
    mocked_objective_function = mocker.Mock()
    with pytest.raises(KeyError):
        ScipyOptimizer(mocked_objective_function,
                       method="Dwayne - The Rock - Johnson")


def get_expected_options(**additional_options):
    default_options = {"method": "BFGS",
                       "tol": 1e-6,
                       "param_init_bounds": [-10000, 10000]}
    default_options.update(additional_options)
    return default_options


def test_can_set_options_via_constructor(mocker):
    mock_obj_fn = \
        mocker.Mock(side_effect=lambda individual: individual.param)

    # testing default options
    opt = ScipyOptimizer(mock_obj_fn)
    assert opt.options == get_expected_options()

    # testing adding extra options
    opt = ScipyOptimizer(mock_obj_fn,
                         options={"maxiter": 0})
    assert opt.options == get_expected_options(options={"maxiter": 0})

    # testing setting default options
    opt = ScipyOptimizer(mock_obj_fn,
                         method="lm",
                         param_init_bounds=[-1, 1],
                         tol=1e-8)
    assert opt.options == get_expected_options(
        method="lm",
        param_init_bounds=[-1, 1],
        tol=1e-8
    )


def test_can_set_options_via_property(mocker):
    mock_obj_fn = \
        mocker.Mock(side_effect=lambda individual: individual.param)
    opt = ScipyOptimizer(mock_obj_fn)

    # testing default options
    assert opt.options == get_expected_options()

    # testing adding extra options
    opt_options = {"options": {"maxiter": 0}}
    opt.options = opt_options
    assert opt.options == get_expected_options(options={"maxiter": 0})

    # testing removing extra options/setting default options
    opt_options = {"method": "lm",
                   "param_init_bounds": [-1, 1],
                   "tol": 1e-8}
    opt.options = opt_options
    assert opt.options == get_expected_options(
        method="lm",
        param_init_bounds=[-1, 1],
        tol=1e-8
    )


@pytest.mark.parametrize("method", ["Nelder-Mead", "lm"])
# using Nelder-Mead and lm to test minimize and root respectively
def test_set_param_bounds_and_clo_options_affect_clo(mocker, method):
    mocked_fitness_function = \
        mocker.Mock(side_effect=lambda individual: individual.param)

    dummy_individual = DummyLocalOptIndividual()

    opt_options = {"method": method,  # TODO have to set this or else it will get overwritten by BFGS on update, I don't like this
                   "tol": 1e-8,
                   "options": {"maxiter": 1000,
                               "fatol": 1e-8,
                               "xatol": 1e-8,
                               "adaptive": False}}

    expected_options = deepcopy(opt_options)
    expected_options["method"] = method
    expected_options["args"] = dummy_individual
    expected_options["jac"] = False

    # returns x=0 if kwargs != expected_options and x=1 vice versa
    def mocked_optimize(*_, **kwargs):
        return OptimizeResult(x=[int(kwargs == expected_options)])

    mocker.patch.object(optimize, "minimize", side_effect=mocked_optimize)
    mocker.patch.object(optimize, "root", side_effect=mocked_optimize)

    opt = ScipyOptimizer(mocked_fitness_function,
                         method=method)

    opt(dummy_individual)
    # default options should != opt_options
    assert dummy_individual.param == 0

    opt.options = opt_options
    opt(dummy_individual)
    # we set opt.options, so options should == opt_options
    assert dummy_individual.param == 1


@pytest.mark.parametrize("method", MINIMIZE_SET)
def test_optimize_params_minimize_without_gradient(mocker, method):
    fitness_function = mocker.create_autospec(FitnessFunction)
    fitness_function.side_effect = lambda individual: 1 + individual.param ** 2

    opt = ScipyOptimizer(fitness_function,
                         param_init_bounds=[5, 5],
                         method=method)

    individual = DummyLocalOptIndividual()
    opt(individual)
    assert fitness_function(individual) == pytest.approx(1, rel=0.05)


class GradientFitnessFunction(GradientMixin, FitnessFunction):
    def __call__(self, individual):
        pass

    def get_fitness_and_gradient(self, individual):
        pass


@pytest.mark.parametrize("method", MINIMIZE_SET)
def test_optimize_params_minimize_with_gradient(mocker, method):
    fitness_function = mocker.create_autospec(GradientFitnessFunction)
    fitness_function.side_effect = lambda individual: 1 + individual.param ** 2
    fitness_function.get_fitness_and_gradient = \
        lambda individual: (1 + individual.param ** 2, 2 * individual.param)

    opt = ScipyOptimizer(fitness_function,
                         param_init_bounds=[5, 5],
                         method=method)

    individual = DummyLocalOptIndividual()
    opt(individual)
    assert fitness_function(individual) == pytest.approx(1, rel=0.05)


class JacobianVectorFitnessFunction(VectorGradientMixin, VectorBasedFunction):
    def evaluate_fitness_vector(self, individual):
        pass

    def get_fitness_vector_and_jacobian(self, individual):
        pass


@pytest.mark.parametrize("method", ROOT_SET)
def test_optimize_params_root_without_jacobian(mocker, method):
    fitness_function = mocker.create_autospec(VectorBasedFunction)
    fitness_function.evaluate_fitness_vector = lambda x: 1 + np.abs([x.param])

    opt = ScipyOptimizer(fitness_function,
                         param_init_bounds=[5, 5],
                         method=method)

    individual = DummyLocalOptIndividual()
    opt(individual)
    opt_indv_fitness = fitness_function.evaluate_fitness_vector(individual)
    assert opt_indv_fitness[0] == pytest.approx(1, rel=0.05)


@pytest.mark.parametrize("method", ROOT_SET)
def test_optimize_params_root_with_jacobian(mocker, method):
    fitness_function = mocker.create_autospec(JacobianVectorFitnessFunction)
    fitness_function.evaluate_fitness_vector = lambda x: 1 + np.abs([x.param])
    fitness_function.get_fitness_vector_and_jacobian = \
        lambda x: (1 + np.abs([x.param]), np.sign([x.param]))

    opt = ScipyOptimizer(fitness_function,
                         param_init_bounds=[5, 5],
                         method=method)

    individual = DummyLocalOptIndividual()
    opt(individual)
    opt_indv_fitness = fitness_function.evaluate_fitness_vector(individual)
    assert opt_indv_fitness[0] == pytest.approx(1, rel=0.05)


@pytest.mark.parametrize("method", ROOT_SET)
def test_optimize_params_too_many_params(mocker, method):
    fitness_function = mocker.create_autospec(VectorBasedFunction)
    fitness_function.evaluate_fitness_vector = lambda x: 1 + np.square(x.param)

    opt = ScipyOptimizer(fitness_function,
                         method=method,
                         param_init_bounds=[-1, -1])

    individual = BloatedOptIndividual()

    assert opt.options["method"] == method

    # root method will error out,
    # should revert to minimize method and optimize
    opt(individual)
    np.testing.assert_array_equal(
        fitness_function.evaluate_fitness_vector(individual),
        np.array([1.0, 1.0, 1.0])
    )

    # make sure method didn't stay as minimize method
    assert opt.options["method"] == method
