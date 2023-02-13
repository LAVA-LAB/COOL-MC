
"""
This module contains the ConstantDefinitionParser.
"""
from typing import List, Tuple, Union
import numpy as np


class ConstantDefinitionParser():
    """
    The ConstantDefinitionParser parse the constant ranges for the range plots.
    """
    @staticmethod
    def constant_range_to_tuple(constant_range: str):
        """The constant range string to constant range tuple

        Args:
            constant_range (str): constant range

        Raises:
            ValueError: if range format is wrong.

        Returns:
            Tuple: Tuple(lower bound, step size, upper bound)
        """
        assert isinstance(constant_range, str)
        range_list = []
        parts = constant_range.split(';')
        if len(parts) != 3:
            raise ValueError('Range Format is wrong...')
        for idx, _ in enumerate(parts):
            try:
                range_list.append(int(parts[idx]))
            except:
                range_list.append(float(parts[idx]))
        range_tuple = tuple(range_list)
        assert isinstance(range_tuple, tuple)
        return range_tuple

    @staticmethod
    def get_range_state_constant(state_constant_assignment) -> Union[None, str]:
        """Get the name of the range state constant.

        Args:
            state_constant_assignment (str): state_constant assignment

        Returns:
            str: range state constant name
        """
        assert isinstance(state_constant_assignment, str)
        parts = state_constant_assignment.split(';')
        for part in parts:
            if part.count('[') == 1:
                sub_parts = part.split('=')
                range_state_constant_name = str(sub_parts[0])
                assert isinstance(range_state_constant_name, str)
                return range_state_constant_name
        return None

    @staticmethod
    def parse_constant_definition(raw_constant_definition: str) -> Tuple[List[str], Tuple[int], str]:
        """Parse the constant definition

        Args:
            raw_constant_definition (str): Raw constant definition string.

        Raises:
            ValueError: Raises error, if the upper bound is smaller than the lower bound.

        Returns:
            Tuple: Tuple of all constant assignments, range tuple, and range constant name
        """
        assert isinstance(raw_constant_definition, str)
        all_constant_definitions = []
        state_constant = ConstantDefinitionParser.get_range_state_constant(
            raw_constant_definition)
        start_idx = raw_constant_definition.find('[')
        end_idx = raw_constant_definition.find(']')
        constant_range = raw_constant_definition[start_idx+1:end_idx]
        tup = ConstantDefinitionParser.constant_range_to_tuple(constant_range)
        if tup[2] < tup[0]:
            raise ValueError("End has to be greater than start")
        for i in np.arange(tup[0], tup[2], tup[1]):
            constant_definition = str(raw_constant_definition)
            to_replace = constant_definition[start_idx:(end_idx+1)]
            constant_definition = constant_definition.replace(
                to_replace, str(i))
            all_constant_definitions.append(constant_definition)
        assert isinstance(all_constant_definitions, list)
        assert isinstance(tup, tuple)
        assert isinstance(state_constant, str)
        return all_constant_definitions, tup, state_constant
