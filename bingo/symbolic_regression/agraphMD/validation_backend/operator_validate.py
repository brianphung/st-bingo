from bingo.symbolic_regression.agraphMD.operator_definitions import *


# NOTE (David Randall): it is assumed node, param1, param2, param3, are all integers
def validate_operator(node, param1, param2, param3, dimensions):
    return VALIDATE_MAP[node](param1, param2, param3, dimensions)


def _integer_validate(param1, param2, param3, dimensions):
    return param2 >= 0 and param3 >= 0, (param2, param3)


def _load_validate(param1, param2, param3, dimensions):
    return param1 >= -1 and param2 >= 0 and param3 >= 0, (param2, param3)


def _addition_like_validate(param1, param2, param3, dimensions):
    return tuple(dimensions[param1]) == tuple(dimensions[param2]), tuple(dimensions[param1])


def _multiply_validate(param1, param2, param3, dimensions):
    dim_1, dim_2 = tuple(dimensions[param1]), tuple(dimensions[param2])
    # scalar * scalar
    if dim_1 == (0, 0) and dim_2 == (0, 0):
        return True, (0, 0)
    # scalar * matrix
    elif dim_1 == (0, 0) and dim_2[0] > 0 and dim_2[1] > 0:
        return True, dim_2
    # matrix * scalar
    elif dim_2 == (0, 0) and dim_1[0] > 0 and dim_1[1] > 0:
        return True, dim_1
    # matrix * matrix
    elif dim_1[1] == dim_2[0]:
        return True, (dim_1[0], dim_2[1])
    else:
        return False, (0, 0)


def _not_implemented_validate(param1, param2, param3, dimensions):
    return False, (-1, -1)


VALIDATE_MAP = {INTEGER: _integer_validate,
                VARIABLE: _load_validate,
                CONSTANT: _load_validate,
                ADDITION: _addition_like_validate,
                SUBTRACTION: _addition_like_validate,
                MULTIPLICATION: _multiply_validate,
                DIVISION: _not_implemented_validate,
                SIN: _not_implemented_validate,
                COS: _not_implemented_validate,
                SINH: _not_implemented_validate,
                COSH: _not_implemented_validate,
                EXPONENTIAL: _not_implemented_validate,
                LOGARITHM: _not_implemented_validate,
                POWER: _not_implemented_validate,
                ABS: _not_implemented_validate,
                SQRT: _not_implemented_validate,
                SAFE_POWER: _not_implemented_validate}
# TODO dot product, another operator???
