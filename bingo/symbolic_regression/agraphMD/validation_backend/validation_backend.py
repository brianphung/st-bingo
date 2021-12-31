import numpy as np

from bingo.symbolic_regression.agraphMD.validation_backend.operator_validate import validate_operator


def validate(stack, output_dim):
    dimensions = np.empty((len(stack), 2))
    for i, (node, param1, param2, param3) in enumerate(stack):
        valid, dimensions[i] = validate_operator(node, param1, param2, param3, dimensions)
        if not valid:
            return False
    if tuple(dimensions[-1]) != output_dim:
        return False
    return True
