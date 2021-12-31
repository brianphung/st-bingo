"""Generator of acyclic graph individuals.

This module contains the implementation of the generation of random acyclic
graph individuals.
"""
import numpy as np

from .agraphMD import AGraphMD
from .validation_backend import validation_backend
from ...chromosomes.generator import Generator
from ...util.argument_validation import argument_validation


class AGraphGeneratorMD(Generator):
    """Generates acyclic graph individuals

    Parameters
    ----------
    agraph_size : int
                  command array size of the generated acyclic graphs
    component_generator : agraph.ComponentGenerator
                          Generator of stack components of agraphs
    """
    @argument_validation(agraph_size={">=": 1})
    def __init__(self, agraph_size, component_generator, output_dim, use_simplification=False):
        self.agraph_size = agraph_size
        self.component_generator = component_generator
        self._output_dim = output_dim
        self._use_simplification = use_simplification
        self._backend_generator_function = self._python_generator_function

    def __call__(self):
        """Generates random agraph individual.

        Fills stack based on random commands from the component generator.

        Returns
        -------
        Agraph
            new random acyclic graph individual
        """
        individual = self._backend_generator_function()
        individual.command_array = self._create_command_array()
        return individual

    def _python_generator_function(self):
        return AGraphMD(use_simplification=self._use_simplification)

    def _create_command_array(self):
        attempts = 0
        potential_command_array = self._generate_potential_command_array()
        while not validation_backend.validate(potential_command_array, self._output_dim):
            if attempts >= 100:
                raise RuntimeWarning("Could not generate valid agraph within 100 attempts")  # TODO test
            potential_command_array = self._generate_potential_command_array()
        return potential_command_array

    def _generate_potential_command_array(self):
        command_array = np.empty((self.agraph_size, 4), dtype=int)
        for i in range(self.agraph_size):
            command_array[i] = self.component_generator.random_command(i)
        return command_array
