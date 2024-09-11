# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Convenience function to load RZXGate based templates.
"""

from enum import Enum
from typing import List, Dict

from qiskit.circuit.library.templates import rzx


def rzx_templates(template_list: List[str] = None) -> Dict:
    """Convenience function to get the cost_dict and templates for template matching.

    Args:
        template_list: List of instruction names.

    Returns:
        Decomposition templates and cost values.
    """

    class RZXTemplateMap(Enum):
        """Mapping of instruction name to decomposition template."""

        ZZ1 = rzx.rzx_zz1()
        ZZ2 = rzx.rzx_zz2()
        ZZ3 = rzx.rzx_zz3()
        YZ = rzx.rzx_yz()
        XZ = rzx.rzx_xz()
        CY = rzx.rzx_cy()

    if template_list is None:
        template_list = ["zz1", "zz2", "zz3", "yz", "xz", "cy"]

    templates = [RZXTemplateMap[gate.upper()].value for gate in template_list]
    cost_dict = {"rzx": 0, "cx": 6, "rz": 0, "sx": 1, "p": 0, "h": 1, "rx": 1, "ry": 1}

    rzx_dict = {"template_list": templates, "user_cost_dict": cost_dict}

    return rzx_dict
