################################
Import and export using QASM 2.0
################################

This guide shows you how to import and export quantum circuits using ``QASM 2.0``.

QASM 2.0 is short for Quantum Assembly Language version 2.0. It's a programming language specifically
designed for representing universal quantum computing using the circuit model.

You can find more information in the  `OpenQASM 2.0 repository <https://github.com/openqasm/openqasm/tree/OpenQASM2.x>`_.

Design a quantum circuit
========================

In this section, you'll build a four-qubit quantum circuit using only CX (CNOT), H (Hadamard), and RZ gates.
This circuit includes a custom quantum gate, created using the :meth:`~qiskit.QuantumCircuit.to_gate` method.
The :meth:`~qiskit.QuantumCircuit.to_gate` method allows us to convert a part of the quantum circuit into a single gate,
which acts like a building block encapsulating the circuit's functionality.

.. plot::
    :include-source:

    from qiskit import QuantumCircuit
    import numpy as np

    custom_gate = QuantumCircuit(1)
    custom_gate.h(0)
    custom_gate.rz(np.pi,0)
    custom_gate.h(0)

    gate = custom_gate.to_gate() # Converts the custom_gate() into a gate.
    gate.name = 'custom X'

    qc = QuantumCircuit(4,4)
    qc.h(0)
    qc.cx(0,1)
    qc.cx(1,2)
    qc.cx(2,3)
    qc.append(gate,[1])  # Adds a custom gate to qubit 1
    qc.barrier()
    qc.measure(range(4),range(4))
    qc.draw("mpl")


Export quantum circuits to OpenQASM 2
=====================================


Converting a quantum circuit to OpenQASM 2 is supported through :meth:`QuantumCircuit.qasm`. By default, it returns a string of QASM code.

.. testcode::

    qc.qasm()

.. testoutput::

    'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate custom_X q0 { h q0; rz(pi) q0; h q0; }\nqreg q[4];\ncreg c[4];\nh q[0];\ncx q[0],q[1];\ncx q[1],q[2];\ncx q[2],q[3];\ncustom_X q[1];\nbarrier q[0],q[1],q[2],q[3];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\nmeasure q[3] -> c[3];\n'

You can print a formatted string by setting the Boolean `formatted` parameter to `True` (default value is `False`).

.. testcode::

    qc.qasm(formatted=True)

.. testoutput::

    OPENQASM 2.0;
    include "qelib1.inc";
    gate custom_X q0 { h q0; rz(pi) q0; h q0; }
    qreg q[4];
    creg c[4];
    h q[0];
    cx q[0],q[1];
    cx q[1],q[2];
    cx q[2],q[3];
    custom_X q[1];
    barrier q[0],q[1],q[2],q[3];
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    measure q[2] -> c[2];
    measure q[3] -> c[3];


You can even save the code in a QASM file by using the ``filename`` parameter.
This parameter should be a string where you specify the name and follow it with the ``.qasm`` extension.


.. testcode::

    qc.qasm(filename='example.qasm')

.. testoutput::

    'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate custom_X q0 { h q0; rz(pi) q0; h q0; }\nqreg q[4];\ncreg c[4];\nh q[0];\ncx q[0],q[1];\ncx q[1],q[2];\ncx q[2],q[3];\ncustom_X q[1];\nbarrier q[0],q[1],q[2],q[3];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\nmeasure q[3] -> c[3];\n'




Import OpenQASM 2 to a quantum circuit
======================================

Qiskit has a specific module called qiskit-qasm2 that imports OpenQASM 2.0 files.

.. note::
    You can install the module using the following command

    ``pip install qiskit-qasm2``


There are two methods available for parsing OpenQASM 2 programs. One method can be used to read a string, and the other method can be used to read a QASM file.
If you want to parse an OpenQASM 2 program from a string into a :class:`QuantumCircuit` you should use :meth:`qiskit.qasm2.loads`.


.. testcode::

    from qiskit import qasm2

    example = 'OPENQASM 2.0;\ninclude "qelib1.inc";\ngate custom_X q0 { h q0; rz(pi) q0; h q0; }\nqreg q[4];\ncreg c[4];\nh q[0];\ncx q[0],q[1];\ncx q[1],q[2];\ncx q[2],q[3];\ncustom_X q[1];\nbarrier q[0],q[1],q[2],q[3];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\nmeasure q[3] -> c[3];\n'
    qc = qasm2.loads(example)
    qc.draw("mpl")


.. plot::

    from qiskit import QuantumCircuit
    import numpy as np

    custom_gate = QuantumCircuit(1) 
    custom_gate.h(0)
    custom_gate.rz(np.pi,0)
    custom_gate.h(0)
    gate = custom_gate.to_gate()
    gate.name = 'custom X'

    qc = QuantumCircuit(4,4)
    qc.h(0)
    qc.cx(0,1)
    qc.cx(1,2)
    qc.cx(2,3)
    qc.append(gate,[1])
    qc.barrier()
    qc.measure(range(4),range(4))
    qc.draw("mpl")

If you have an OpenQASM 2 program in a file, you should use the :meth:`qiskit.qasm2.load` method.


.. testcode::

    qc = qasm2.loads('example.qasm')
    qc.draw("mpl")


.. plot::

    from qiskit import QuantumCircuit
    import numpy as np

    custom_gate = QuantumCircuit(1) 
    custom_gate.h(0)
    custom_gate.rz(np.pi,0)
    custom_gate.h(0)
    gate = custom_gate.to_gate()
    gate.name = 'custom X'

    qc = QuantumCircuit(4,4)
    qc.h(0)
    qc.cx(0,1)
    qc.cx(1,2)
    qc.cx(2,3)
    qc.append(gate,[1])
    qc.barrier()
    qc.measure(range(4),range(4))
    qc.draw("mpl")