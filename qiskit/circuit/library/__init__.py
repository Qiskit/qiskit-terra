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


r"""
===============================================
Circuit Library (:mod:`qiskit.circuit.library`)
===============================================

.. currentmodule:: qiskit.circuit.library

The circuit library is a collection of valuable circuits and building blocks. We call these valuable
for different reasons. For instance, they can be used as building blocks for algorithms, serve as 
benchmarks, or they are circuits are conjectured to be difficult to simulate classically.

Elements in the circuit library are provided as :class:`.QuantumCircuit`\ s or 
:class:`~.circuit.Instruction`\ s, allowing them to be easily investigated or plugged into other 
circuits. This enables fast prototyping and circuit design circuit at higher levels of abstraction.

For example:

.. plot::
   :alt: Circuit diagram output by the previous code.
   :include-source:

   from qiskit.circuit import QuantumCircuit
   from qiskit.circuit.library import PauliEvolutionGate
   from qiskit.quantum_info import SparsePauliOp

   hamiltonian = SparsePauliOp(["ZZI", "IZZ", "IXI"], coeffs=[1, 1, -1])
   gate = PauliEvolutionGate(hamiltonian)

   circuit = QuantumCircuit(hamiltonian.num_qubits)
   circuit.append(gate, circuit.qubits)

   circuit.draw("mpl")

This library is organized in different sections:

   * :ref:`Standard gates <standard-gates>`, :ref:`directives <standard-directives>`,
     and :ref:`operations <standard-operations>`
   * :ref:`Generalized gates <generalized-gates>`
   * :ref:`Arithmetic operations <arithmetic>`
   * :ref:`Basis changes <basis-change>`
   * :ref:`Boolean logic <boolean-logic>`
   * :ref:`Data encoding <data-encoding>` and :ref:`preparation <data-preparation>`
   * :ref:`Particular operations <particular>`
   * :ref:`N-local circuits <n-local>`
   * :ref:`Template circuits <template>`

We distinguish into different categories of operations:

Standard gates
   These are fundamental quantum gates, a subset of which typically forms a basis gate
   set on a quantum computer. These are unitary operations represented as :class:`.Gate`. 
   The library also provides standard compiler directives (a :class:`.Barrier`) and non-unitary
   operations (like :class:`.Measure`).

Abstract operations
   This describes operations that are defined by a mathematical action, but can be implemented with 
   different decompositions. For example, a multi-controlled X gate flips the target qubit if all 
   control qubits are :math:`|1\rangle`, and there are a variety of concrete circuits implementing 
   this operation using lower-level gates. Such abstract operations are represented as :class:`.Gate` 
   or  :class:`~.circuit.Instruction`. This allows building the circuit without choosing a concrete 
   implementation of each block and, finally, let the compiler (or you as user) choose the optimal 
   decomposition. For example:

   .. plot::
      :include-source:

      from qiskit.circuit.library import MCXGate
      mcx = MCXGate(4)

      from qiskit import QuantumCircuit
      circuit = QuantumCircuit(5)
      circuit.append(mcx, [0, 1, 4, 2, 3])
      circuit.draw("mpl")

   When this circuit is transpiled the circuit context is taken into account. For example, if idle
   qubits are available they can be used to obtain a shallower circuit::

     from qiskit import transpile

     small_circuit = QuantumCircuit(5)  # here we have no idle qubits
     small_circuit.append(mcx, [0, 1, 4, 2, 3])
     small_tqc = transpile(small_circuit, basis_gates=["u", "cx"])
     print("No aux:", small_tqc.count_ops())  

     large_circuit = QuantumCircuit(10)  # now we will have 5 idle qubits
     large_circuit.append(mcx, [0, 1, 4, 2, 3])
     large_tqc = transpile(large_circuit, basis_gates=["u", "cx"])
     print("With aux:", large_tqc.count_ops())

   Which prints:

   .. parsed-literal::

      No aux: OrderedDict([('u', 41), ('cx', 36)])
      With aux: OrderedDict([('u', 24), ('cx', 18)])

Structural operations
   These operations have a unique decomposition. As the compiler does not need to reason about
   them on a higher level, they are implemented as function that return a :class:`.QuantumCircuit`
   object. For example:

   .. plot::
      :include-source:

      from qiskit.circuit.library import real_amplitudes

      ansatz = real_amplitudes(5, entanglement="pairwise")
      ansatz.draw("mpl")

      
.. _standard-gates:
     
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

     
The function :func:`.get_standard_gate_name_mapping` allows you to see the available standard gates 
and operations.

.. autofunction:: get_standard_gate_name_mapping

.. autosummary::
   :toctree: ../stubs/

   C3SXGate
   C3XGate
   C4XGate
   CCXGate
   CCZGate
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
   DCXGate
   ECRGate
   GlobalPhaseGate
   HGate
   IGate
   iSwapGate
   MSGate
   PhaseGate
   RC3XGate
   RCCXGate
   RGate
   RXGate
   RXXGate
   RYGate
   RYYGate
   RZGate
   RZXGate
   RZZGate
   SGate
   SdgGate
   SwapGate
   SXGate
   SXdgGate
   TGate
   TdgGate
   UGate
   U1Gate
   U2Gate
   U3Gate
   XGate
   XXMinusYYGate
   XXPlusYYGate
   YGate
   ZGate

   
.. _standard-directives:
   
Standard Directives
===================

Directives are operations to the quantum stack that are meant to be interpreted by the backend or
the transpiler. In general, the transpiler or backend might optionally ignore them if there is no
implementation for them.

* :class:`~qiskit.circuit.Barrier`


.. _standard-operations:

Standard Operations
===================

Operations are non-reversible changes in the quantum state of the circuit.

* :class:`~qiskit.circuit.Measure`
* :class:`~qiskit.circuit.Reset`


.. _generalized-gates:

Generalized Gates
=================

This module extends the standard gates by a broader collection of basic gates. This includes
gates that are variadic, meaning that the number of qubits depends on the input.
For example::

    from qiskit.circuit.library import DiagonalGate

    diagonal = DiagonalGate([1, 1j])
    print(diagonal.num_qubits)

    diagonal = DiagonalGate([1, 1, 1, -1])
    print(diagonal.num_qubits)

which prints:

.. code-block:: text

    1
    2

.. autosummary::
   :toctree: ../stubs/

   DiagonalGate
   PermutationGate
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

The above objects derive :class:`.Gate` or :class:`~.circuit.Instruction`, which allows the 
compiler to reason about them on an abstract level. We therefore suggest using these instead 
of the following which derive :class:`.QuantumCircuit` and are eagerly constructed.

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   Diagonal
   MCMT
   MCMTVChain
   Permutation
   GMS
   GR
   GRX
   GRY
   GRZ

.. _boolean-logic:

Boolean Logic Circuits
======================

These :class:`.Gate`\ s implement boolean logic operations, such as the logical
or of a set of qubit states.

.. autosummary::
   :toctree: ../stubs/

   AndGate
   OrGate
   BitwiseXorGate
   random_bitwise_xor
   InnerProductGate

The above objects derive :class:`.Gate` (or return this type), which allows the 
compiler to reason about them on an abstract level. We therefore suggest using these instead 
of the following which derive :class:`.QuantumCircuit` and are eagerly constructed.

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   AND
   OR
   XOR
   InnerProduct

.. _basis-change:
   
Basis Change Circuits
=====================

These gates allow basis transformations of the qubit states. For example,
in the case of the Quantum Fourier Transform (QFT), it transforms between
the computational basis and the Fourier basis.

.. autosummary::
   :toctree: ../stubs/

   QFTGate

The above objects derive :class:`.Gate`, which allows the 
compiler to reason about them on an abstract level. We therefore suggest using these instead 
of the following which derive :class:`.QuantumCircuit` and are eagerly constructed.

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   QFT


.. _arithmetic:

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

   ModularAdderGate
   HalfAdderGate
   FullAdderGate

The above objects derive :class:`.Gate`, which allows the 
compiler to reason about them on an abstract level. We therefore suggest using these instead 
of the following which derive :class:`.QuantumCircuit` and are eagerly constructed.

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   DraperQFTAdder
   CDKMRippleCarryAdder
   VBERippleCarryAdder
   WeightedAdder


Multipliers
-----------

.. autosummary::
   :toctree: ../stubs/

   MultiplierGate

The above object derives :class:`.Gate`, which allows the 
compiler to reason about it on an abstract level. We therefore suggest using these instead 
of the following which derive :class:`.QuantumCircuit` and are eagerly constructed.

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   HRSCumulativeMultiplier
   RGQFTMultiplier

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


.. _particular:   

Particular Quantum Circuits
===========================

The following gates and quantum circuits define specific
quantum circuits of interest:

.. autosummary::
   :toctree: ../stubs/

   fourier_checking
   hidden_linear_function
   iqp
   random_iqp
   quantum_volume
   phase_estimation
   grover_operator
   unitary_overlap


While the above are functions returning circuits, the following 
classes directly derive :class:`.QuantumCircuits`. For better performance, however,
we suggest using above functions.  

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


.. _n-local:   

N-local circuits
================

The following functions return a parameterized :class:`.QuantumCircuit` to use as ansatz in
a broad set of variational quantum algorithms. 

For example, we can build a variational circuit:

.. plot::
   :include-source:
   :context:

   from qiskit.circuit.library import efficient_su2

   num_qubits = 4
   ansatz = efficient_su2(num_qubits, entanglement="pairwise")
   ansatz.draw("mpl")

and combine it with:

.. plot::
   :include-source:
   :context:

   from qiskit.circuit.library import zz_feature_map

   circuit = zz_feature_map(num_qubits)
   circuit.barrier()
   circuit.compose(ansatz, inplace=True)

   circuit.draw("mpl")


.. autosummary::
   :toctree: ../stubs/

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


.. _data-encoding:
   
Data encoding circuits
======================

The following functions return a parameterized :class:`.QuantumCircuit` to use as data
encoding circuits in a series of variational quantum algorithms:

.. autosummary::
   :toctree: ../stubs/

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


.. _data-preparation:
   
Data preparation circuits
=========================

The following operations are used for state preparation:

.. autosummary::
   :toctree: ../stubs/

   StatePreparation
   Initialize

.. _template:   

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
