# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Second-order Pauli-Z expansion circuit."""

from typing import Callable, List, Union, Optional, Dict, Tuple
import numpy as np
from .pauli_feature_map import PauliFeatureMap


class ZZFeatureMap(PauliFeatureMap):
    r"""Second-order Pauli-Z evolution circuit.

    For 3 qubits and 1 repetition and linear entanglement the circuit is represented by:

    .. parsed-literal::

        ┌───┐┌─────────────────┐
        ┤ H ├┤ U1(2.0*φ(x[0])) ├──■────────────────────────────■────────────────────────────────────
        ├───┤├─────────────────┤┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[1])) ├┤ X ├┤ U1(2.0*φ(x[0],x[1])) ├┤ X ├──■────────────────────────────■──
        ├───┤├─────────────────┤└───┘└──────────────────────┘└───┘┌─┴─┐┌──────────────────────┐┌─┴─┐
        ┤ H ├┤ U1(2.0*φ(x[2])) ├──────────────────────────────────┤ X ├┤ U1(2.0*φ(x[1],x[2])) ├┤ X ├
        └───┘└─────────────────┘                                  └───┘└──────────────────────┘└───┘

    where :math:`\varphi` is a classical non-linear function, which defaults to :math:`\varphi(x) = x`
    if and :math:`\varphi(x,y) = (\pi - x)(\pi - y)`.

    Examples:

        >>> from qiskit.circuit.library import ZZFeatureMap
        >>> prep = ZZFeatureMap(2, reps=1)
        >>> print(prep)
             ┌───┐┌──────────────┐
        q_0: ┤ H ├┤ U1(2.0*x[0]) ├──■───────────────────────────────────────■──
             ├───┤├──────────────┤┌─┴─┐┌─────────────────────────────────┐┌─┴─┐
        q_1: ┤ H ├┤ U1(2.0*x[1]) ├┤ X ├┤ U1(2.0*(pi - x[0])*(pi - x[1])) ├┤ X ├
             └───┘└──────────────┘└───┘└─────────────────────────────────┘└───┘

        >>> from qiskit.circuit.library import EfficientSU2
        >>> classifier = ZZFeatureMap(3) + EfficientSU2(3)
        >>> classifier.num_parameters
        15
        >>> classifier.parameters  # 'x' for the data preparation, 'θ' for the SU2 parameters
        ParameterView([
            ParameterVectorElement(x[0]), ParameterVectorElement(x[1]),
            ParameterVectorElement(x[2]), ParameterVectorElement(θ[0]),
            ParameterVectorElement(θ[1]), ParameterVectorElement(θ[2]),
            ParameterVectorElement(θ[3]), ParameterVectorElement(θ[4]),
            ParameterVectorElement(θ[5]), ParameterVectorElement(θ[6]),
            ParameterVectorElement(θ[7]), ParameterVectorElement(θ[8]),
            ParameterVectorElement(θ[9]), ParameterVectorElement(θ[10]),
            ParameterVectorElement(θ[11]), ParameterVectorElement(θ[12]),
            ParameterVectorElement(θ[13]), ParameterVectorElement(θ[14]),
            ParameterVectorElement(θ[15]), ParameterVectorElement(θ[16]),
            ParameterVectorElement(θ[17]), ParameterVectorElement(θ[18]),
            ParameterVectorElement(θ[19]), ParameterVectorElement(θ[20]),
            ParameterVectorElement(θ[21]), ParameterVectorElement(θ[22]),
            ParameterVectorElement(θ[23])
        ])
        >>> classifier.count_ops()
        OrderedDict([('ZZFeatureMap', 1), ('EfficientSU2', 1)])
    """

    def __init__(
        self,
        feature_dimension: int,
        reps: int = 2,
        entanglement: Union[
            str,
            Dict[int, List[Tuple[int]]],
            Callable[[int], Union[str, Dict[int, List[Tuple[int]]]]],
        ] = "full",
        data_map_func: Optional[Callable[[np.ndarray], float]] = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "ZZFeatureMap",
    ) -> None:
        """Create a new second-order Pauli-Z expansion.

        Args:
            feature_dimension: Number of features.
            reps: The number of repeated circuits, has a min. value of 1.
            entanglement: Specifies the entanglement structure. Refer to
                :class:`~qiskit.circuit.library.PauliFeatureMap` for detail.
            data_map_func: A mapping function for data x.
            parameter_prefix: The prefix used if default parameters are generated.
            insert_barriers: If True, barriers are inserted in between the evolution instructions
                and hadamard layers.

        Raises:
            ValueError: If the feature dimension is smaller than 2.
        """
        if feature_dimension < 2:
            raise ValueError(
                "The ZZFeatureMap contains 2-local interactions and cannot be "
                f"defined for less than 2 qubits. You provided {feature_dimension}."
            )

        super().__init__(
            feature_dimension=feature_dimension,
            reps=reps,
            entanglement=entanglement,
            paulis=["Z", "ZZ"],
            data_map_func=data_map_func,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            name=name,
        )
