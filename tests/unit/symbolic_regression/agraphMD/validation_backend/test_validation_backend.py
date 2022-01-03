import numpy as np
import pytest

from bingo.symbolic_regression.agraphMD.operator_definitions import *
from bingo.symbolic_regression.agraphMD.validation_backend.validation_backend import validate


@pytest.fixture
def complex_stack():
    return np.array([[VARIABLE, 0, 3, 1],
                     [CONSTANT, 0, 1, 3],
                     [MULTIPLICATION, 0, 1, 0],
                     [CONSTANT, 1, 0, 0],
                     [MULTIPLICATION, 2, 3, 0],
                     [CONSTANT, 2, 3, 3],
                     [ADDITION, 4, 5, 0],
                     [SUBTRACTION, 4, 6, 0]], dtype=int)


@pytest.mark.parametrize("node", [ADDITION, SUBTRACTION])
def test_validate_addition_like_valid(node):
    stack = np.array([[CONSTANT, 0, 1, 2],
                      [VARIABLE, 0, 1, 2],
                      [node, 0, 1, 0]], dtype=int)
    assert validate(stack, (1, 2)) is True


@pytest.mark.parametrize("node", [ADDITION, SUBTRACTION])
def test_validate_addition_like_invalid_different_dims(node):
    stack = np.array([[CONSTANT, 0, 2, 2],
                      [VARIABLE, 0, 1, 2],
                      [node, 0, 1, 0]], dtype=int)
    assert validate(stack, (1, 2)) is False


@pytest.mark.parametrize("op1_dims, op2_dims", [[(0, 0), (0, 0)],  # scalar * scalar
                                                [(1, 3), (0, 0)],  # matrix * scalar
                                                [(0, 0), (1, 3)],  # scalar * matrix
                                                [(1, 3), (3, 2)]])  # matrix * matrix
def test_validate_multiplication_valid(op1_dims, op2_dims):
    stack = np.array([[CONSTANT, 0, *op1_dims],
                      [CONSTANT, 1, *op2_dims],
                      [MULTIPLICATION, 0, 1, 0]], dtype=int)

    output_dim = (op1_dims[0], op2_dims[1])
    if op1_dims == (0, 0):
        output_dim = op2_dims
    elif op2_dims == (0, 0):
        output_dim = op1_dims
    assert validate(stack, output_dim) is True


def test_validate_multiplication_invalid():
    # matrix * matrix with bad dims
    stack = np.array([[CONSTANT, 0, 1, 3],
                      [CONSTANT, 0, 2, 1],
                      [MULTIPLICATION, 0, 1, 0]], dtype=int)
    assert validate(stack, (1, 1)) is False


def test_validate_complex(complex_stack):
    assert validate(complex_stack, (3, 3)) is True


def test_validate_bad_output_dim(complex_stack):
    assert validate(complex_stack, (1, 3)) is False
