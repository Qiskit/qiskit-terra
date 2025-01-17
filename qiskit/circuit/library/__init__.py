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

"""
===============================================
Circuit Library (:mod:`qiskit.circuit.library`)
===============================================

.. currentmodule:: qiskit.circuit.library

The circuit library is a collection of well-studied and valuable circuits, directives, and gates.
We call them valuable for different reasons, for instance they can serve as building blocks for
algorithms or they are circuits that we think are hard to simulate classically.

Each element can be plugged into a circuit using the :meth:`.QuantumCircuit.append`
method and so the circuit library allows users to program at higher levels of abstraction.
For example, to append a multi-controlled CNOT:

.. plot::
   :alt: Circuit diagram output by the previous code.
   :include-source:

   from qiskit.circuit.library import MCXGate
   gate = MCXGate(4)

   from qiskit import QuantumCircuit
   circuit = QuantumCircuit(5)
   circuit.append(gate, [0, 1, 4, 2, 3])
   circuit.draw('mpl')

The library is organized in several sections. The function
:func:`.get_standard_gate_name_mapping` allows you to see the available standard gates and operations.

.. autofunction:: get_standard_gate_name_mapping


Standard gates
==============

These operations are reversible unitary gates and they all subclass
:class:`~qiskit.circuit.Gate`. As a consequence, they all have the methods
:meth:`~qiskit.circuit.Gate.to_matrix`, :meth:`~qiskit.circuit.Gate.power`,
and :meth:`~qiskit.circuit.Gate.control`, which we can generally only apply to unitary operations.

For example:

.. plot::
   :include-source:
   :nofigs:

    from qiskit.circuit.library import XGate
    gate = XGate()
    print(gate.to_matrix())             # X gate
    print(gate.power(1/2).to_matrix())  # √X gate
    print(gate.control(1).to_matrix())  # CX (controlled X) gate

.. code-block:: text

    [[0.+0.j 1.+0.j]
     [1.+0.j 0.+0.j]]
    [[0.5+0.5j 0.5-0.5j]
     [0.5-0.5j 0.5+0.5j]]
    [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 1.+0.j]
     [0.+0.j 0.+0.j 1.+0.j 0.+0.j]
     [0.+0.j 1.+0.j 0.+0.j 0.+0.j]]

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   C3XGate
   C3SXGate
   C4XGate
   CCXGate
   DCXGate
   CHGate
   CPhaseGate
   CRXGate
   CRYGate
   CRZGate
   CSGate
   CSdgGate
   CSwapGate
   CSXGate
   CUGate
   CU1Gate
   CU3Gate
   CXGate
   CYGate
   CZGate
   CCZGate
   ECRGate
   HGate
   IGate
   MSGate
   PhaseGate
   RCCXGate
   RC3XGate
   RGate
   RXGate
   RXXGate
   RYGate
   RYYGate
   RZGate
   RZZGate
   RZXGate
   XXMinusYYGate
   XXPlusYYGate
   SGate
   SdgGate
   SwapGate
   iSwapGate
   SXGate
   SXdgGate
   TGate
   TdgGate
   UGate
   U1Gate
   U2Gate
   U3Gate
   XGate
   YGate
   ZGate
   GlobalPhaseGate


Standard Directives
===================

Directives are operations to the quantum stack that are meant to be interpreted by the backend or
the transpiler. In general, the transpiler or backend might optionally ignore them if there is no
implementation for them.

* :class:`qiskit.circuit.Barrier`

Standard Operations
===================

Operations are non-reversible changes in the quantum state of the circuit.

* :class:`qiskit.circuit.Measure`
* :class:`qiskit.circuit.Reset`

Generalized Gates
=================

These "gates" (many are :class:`~qiskit.circuit.QuantumCircuit` subclasses) allow to
set the amount of qubits involved at instantiation time.


.. plot::
   :include-source:
   :nofigs:

    from qiskit.circuit.library import DiagonalGate

    diagonal = DiagonalGate([1, 1j])
    print(diagonal.num_qubits)

    diagonal = DiagonalGate([1, 1, 1, -1])
    print(diagonal.num_qubits)

.. code-block:: text

    1
    2


.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   Diagonal
   DiagonalGate
   MCMT
   MCMTVChain
   Permutation
   PermutationGate
   GMS
   GR
   GRX
   GRY
   GRZ
   MCMTGate
   MCPhaseGate
   MCXGate
   MCXGrayCode
   MCXRecursive
   MCXVChain
   RVGate
   PauliGate
   LinearFunction
   Isometry
   UnitaryGate
   UCGate
   UCPauliRotGate
   UCRXGate
   UCRYGate
   UCRZGate

Boolean Logic Circuits
======================

These are :class:`~qiskit.circuit.QuantumCircuit` subclasses
that implement boolean logic operations, such as the logical
or of a set of qubit states.


.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   AND
   AndGate
   OR
   OrGate
   XOR
   BitwiseXorGate
   random_bitwise_xor
   InnerProduct
   InnerProductGate


Basis Change Circuits
=====================

These circuits allow basis transformations of the qubit states. For example,
in the case of the Quantum Fourier Transform (QFT), it transforms between
the computational basis and the Fourier basis.

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   QFT
   QFTGate

Arithmetic Circuits
===================

These :class:`~qiskit.circuit.QuantumCircuit`\\ s perform classical arithmetic,
such as addition or multiplication.

Amplitude Functions
-------------------

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   LinearAmplitudeFunction

Functional Pauli Rotations
--------------------------

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   FunctionalPauliRotations
   LinearPauliRotations
   PolynomialPauliRotations
   PiecewiseLinearPauliRotations
   PiecewisePolynomialPauliRotations
   PiecewiseChebyshev

Adders
------

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   DraperQFTAdder
   CDKMRippleCarryAdder
   VBERippleCarryAdder
   WeightedAdder
   ModularAdderGate
   HalfAdderGate
   FullAdderGate

Multipliers
-----------

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   HRSCumulativeMultiplier
   RGQFTMultiplier
   MultiplierGate

Comparators
-----------

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   IntegerComparator

Functions on binary variables
-----------------------------

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   QuadraticForm

Other arithmetic functions
--------------------------

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   ExactReciprocal

Particular Quantum Circuits
===========================

The following gates and quantum circuits define specific
quantum circuits of interest:

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   FourierChecking
   GraphState
   GraphStateGate
   HiddenLinearFunction
   IQP
   QuantumVolume
   PhaseEstimation
   GroverOperator
   PhaseOracle
   PauliEvolutionGate
   HamiltonianGate
   UnitaryOverlap

For circuits that have a well-defined structure it is preferrable
to use the following functions to construct them:

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   fourier_checking
   hidden_linear_function
   iqp
   random_iqp
   quantum_volume
   phase_estimation
   grover_operator
   unitary_overlap


N-local circuits
================

The following functions return a parameterized :class:`.QuantumCircuit` to use as ansatz in
a broad set of variational quantum algorithms:

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   n_local
   efficient_su2
   real_amplitudes
   pauli_two_design
   excitation_preserving
   qaoa_ansatz
   hamiltonian_variational_ansatz
   evolved_operator_ansatz

These :class:`~qiskit.circuit.library.BlueprintCircuit` subclasses are used
as parameterized models (a.k.a. ansatzes or variational forms) in variational algorithms.
They are heavily used in near-term algorithms in e.g. Chemistry, Physics or Optimization.

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   NLocal
   TwoLocal
   PauliTwoDesign
   RealAmplitudes
   EfficientSU2
   EvolvedOperatorAnsatz
   ExcitationPreserving
   QAOAAnsatz


Data encoding circuits
======================

The following functions return a parameterized :class:`.QuantumCircuit` to use as data
encoding circuits in a series of variational quantum algorithms:

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   pauli_feature_map
   z_feature_map
   zz_feature_map

These :class:`~qiskit.circuit.library.BlueprintCircuit` encode classical
data in quantum states and are used as feature maps for classification.

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   PauliFeatureMap
   ZFeatureMap
   ZZFeatureMap


Data preparation circuits
=========================

The following operations are used for state preparation:

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   StatePreparation
   Initialize

Template circuits
=================

Templates are functions that return circuits that compute the identity. They are used at
circuit optimization where matching part of the template allows the compiler
to replace the match with the inverse of the remainder from the template.

In this example, the identity constant in a template is checked:

.. plot::
   :include-source:
   :nofigs:

    from qiskit.circuit.library.templates import template_nct_4b_1
    from qiskit.quantum_info import Operator
    import numpy as np

    template = template_nct_4b_1()

    identity = np.identity(2 ** len(template.qubits), dtype=complex)
    data = Operator(template).data
    np.allclose(data, identity)  # True, template_nct_4b_1 is the identity

NCT (Not-CNOT-Toffoli) template circuits
----------------------------------------

Template circuits for :class:`~qiskit.circuit.library.XGate`,
:class:`~qiskit.circuit.library.CXGate`,
and :class:`~qiskit.circuit.library.CCXGate` (Toffoli) gates.

**Reference:**
Maslov, D. and Dueck, G. W. and Miller, D. M.,
Techniques for the synthesis of reversible Toffoli networks, 2007
http://dx.doi.org/10.1145/1278349.1278355

.. currentmodule:: qiskit.circuit.library.templates.nct
.. autofunction:: template_nct_2a_1
.. autofunction:: template_nct_2a_2
.. autofunction:: template_nct_2a_3
.. autofunction:: template_nct_4a_1
.. autofunction:: template_nct_4a_2
.. autofunction:: template_nct_4a_3
.. autofunction:: template_nct_4b_1
.. autofunction:: template_nct_4b_2
.. autofunction:: template_nct_5a_1
.. autofunction:: template_nct_5a_2
.. autofunction:: template_nct_5a_3
.. autofunction:: template_nct_5a_4
.. autofunction:: template_nct_6a_1
.. autofunction:: template_nct_6a_2
.. autofunction:: template_nct_6a_3
.. autofunction:: template_nct_6a_4
.. autofunction:: template_nct_6b_1
.. autofunction:: template_nct_6b_2
.. autofunction:: template_nct_6c_1
.. autofunction:: template_nct_7a_1
.. autofunction:: template_nct_7b_1
.. autofunction:: template_nct_7c_1
.. autofunction:: template_nct_7d_1
.. autofunction:: template_nct_7e_1
.. autofunction:: template_nct_9a_1
.. autofunction:: template_nct_9c_1
.. autofunction:: template_nct_9c_2
.. autofunction:: template_nct_9c_3
.. autofunction:: template_nct_9c_4
.. autofunction:: template_nct_9c_5
.. autofunction:: template_nct_9c_6
.. autofunction:: template_nct_9c_7
.. autofunction:: template_nct_9c_8
.. autofunction:: template_nct_9c_9
.. autofunction:: template_nct_9c_10
.. autofunction:: template_nct_9c_11
.. autofunction:: template_nct_9c_12
.. autofunction:: template_nct_9d_1
.. autofunction:: template_nct_9d_2
.. autofunction:: template_nct_9d_3
.. autofunction:: template_nct_9d_4
.. autofunction:: template_nct_9d_5
.. autofunction:: template_nct_9d_6
.. autofunction:: template_nct_9d_7
.. autofunction:: template_nct_9d_8
.. autofunction:: template_nct_9d_9
.. autofunction:: template_nct_9d_10
.. currentmodule:: qiskit.circuit.library

Clifford template circuits
--------------------------

Template circuits over Clifford gates.

.. autofunction:: clifford_2_1
.. autofunction:: clifford_2_2
.. autofunction:: clifford_2_3
.. autofunction:: clifford_2_4
.. autofunction:: clifford_3_1
.. autofunction:: clifford_4_1
.. autofunction:: clifford_4_2
.. autofunction:: clifford_4_3
.. autofunction:: clifford_4_4
.. autofunction:: clifford_5_1
.. autofunction:: clifford_6_1
.. autofunction:: clifford_6_2
.. autofunction:: clifford_6_3
.. autofunction:: clifford_6_4
.. autofunction:: clifford_6_5
.. autofunction:: clifford_8_1
.. autofunction:: clifford_8_2
.. autofunction:: clifford_8_3

RZXGate template circuits
-------------------------

Template circuits with :class:`~qiskit.circuit.library.RZXGate`.

.. autofunction:: rzx_yz
.. autofunction:: rzx_xz
.. autofunction:: rzx_cy
.. autofunction:: rzx_zz1
.. autofunction:: rzx_zz2
.. autofunction:: rzx_zz3

"""

from .standard_gates import *
from .templates import *
from ..barrier import Barrier
from ..measure import Measure
from ..reset import Reset


from .blueprintcircuit import BlueprintCircuit
from .generalized_gates import (
    Diagonal,
    DiagonalGate,
    MCMT,
    MCMTVChain,
    Permutation,
    PermutationGate,
    GMS,
    MCMTGate,
    MSGate,
    GR,
    GRX,
    GRY,
    GRZ,
    RVGate,
    PauliGate,
    LinearFunction,
    Isometry,
    UnitaryGate,
    UCGate,
    UCPauliRotGate,
    UCRXGate,
    UCRYGate,
    UCRZGate,
)
from .pauli_evolution import PauliEvolutionGate
from .hamiltonian_gate import HamiltonianGate
from .boolean_logic import (
    AND,
    AndGate,
    OR,
    OrGate,
    XOR,
    BitwiseXorGate,
    random_bitwise_xor,
    InnerProduct,
    InnerProductGate,
)
from .basis_change import QFT, QFTGate
from .arithmetic import (
    ModularAdderGate,
    HalfAdderGate,
    FullAdderGate,
    MultiplierGate,
    FunctionalPauliRotations,
    LinearPauliRotations,
    PiecewiseLinearPauliRotations,
    PiecewisePolynomialPauliRotations,
    PolynomialPauliRotations,
    IntegerComparator,
    WeightedAdder,
    QuadraticForm,
    LinearAmplitudeFunction,
    VBERippleCarryAdder,
    CDKMRippleCarryAdder,
    DraperQFTAdder,
    PiecewiseChebyshev,
    HRSCumulativeMultiplier,
    RGQFTMultiplier,
    ExactReciprocal,
)

from .n_local import (
    n_local,
    NLocal,
    TwoLocal,
    pauli_two_design,
    PauliTwoDesign,
    real_amplitudes,
    RealAmplitudes,
    efficient_su2,
    EfficientSU2,
    hamiltonian_variational_ansatz,
    evolved_operator_ansatz,
    EvolvedOperatorAnsatz,
    excitation_preserving,
    ExcitationPreserving,
    qaoa_ansatz,
    QAOAAnsatz,
)
from .data_preparation import (
    z_feature_map,
    zz_feature_map,
    pauli_feature_map,
    PauliFeatureMap,
    ZFeatureMap,
    ZZFeatureMap,
    StatePreparation,
    Initialize,
)
from .quantum_volume import QuantumVolume, quantum_volume
from .fourier_checking import FourierChecking, fourier_checking
from .graph_state import GraphState, GraphStateGate
from .hidden_linear_function import HiddenLinearFunction, hidden_linear_function
from .iqp import IQP, iqp, random_iqp
from .phase_estimation import PhaseEstimation, phase_estimation
from .grover_operator import GroverOperator, grover_operator
from .phase_oracle import PhaseOracle
from .overlap import UnitaryOverlap, unitary_overlap
from .standard_gates import get_standard_gate_name_mapping
