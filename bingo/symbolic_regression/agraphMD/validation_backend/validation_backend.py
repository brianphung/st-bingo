import numpy as np

from bingo.symbolic_regression.agraphMD.validation_backend.operator_validate import validate_operator, is_scalar_shape


def validate_individual(individual):
    return validate(individual.command_array, individual.input_dims, individual.output_dim, individual.constant_shapes)


def validate(stack, x_dims, output_dim, constant_dims):
    dimensions = np.empty((len(stack), 2))
    for i, (node, param1, param2, param3) in enumerate(stack):
        valid, dimensions[i] = validate_operator(node, param1, param2, param3, dimensions, x_dims, constant_dims)
        if not valid:
            return False
    if tuple(dimensions[-1]) != output_dim:
        return False
    return True


def get_stack_command_dimensions(stack, x_dims, constant_dims):
    dimensions = np.empty((len(stack), 2))
    for i, (node, param1, param2, param3) in enumerate(stack):
        valid, dimensions[i] = validate_operator(node, param1, param2, param3, dimensions, x_dims, constant_dims)
        if not valid:
            raise RuntimeError("stack not valid")
    dimensions = [tuple(dim) for dim in dimensions]
    return dimensions
