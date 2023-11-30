#####################################
Setting transpiler optimization level
#####################################

This guide describes the effects of the ``optimization_level``
parameter of :func:`~qiskit.compiler.transpile`.

``optimization_level`` helps you to optimize your quantum circuit.
This parameter takes an integer which can be a value between 0 and 3,
where the higher the number, the more optimized the result.
You can find more information about this parameter's value and its meaning for
the internals of the :mod:`~.transpiler` in :ref:`working_with_preset_pass_managers`.

Initialize the quantum circuit
==============================

For this example, you will explore the :class:`.CSwapGate` gate,
which is a three-qubit gate which will be transpiled into one and two-qubit gates.

.. plot::
    :include-source:

    from qiskit import QuantumCircuit 
    from qiskit.compiler import transpile
    from qiskit.providers.fake_provider import FakeQuitoV2

    backend = FakeQuitoV2()

    qc = QuantumCircuit(3) # Initialize the quantum circuit with 3 qubits.
    
    qc.cswap(0,1,2) # Add the cswap gate to the quantum circuit.
    
    qc.draw("mpl")


Using backend’s information
===========================

The effect of setting the ``optimization_level`` will differ depending on the backend you are using.
When using :func:`~qiskit.transpile`, make sure you're transpiling for your specific backend. 
This process entails breaking down your circuit into basis gates and considering the physical connections specified in the 
coupling map for two-qubit gates.
Given the presence of noise in the backend, it is crucial to optimize your circuit by adjusting the ``optimization_level`` parameter. 
This will help minimize the number of circuit operations and enhance the overall performance.

When using a backend, you can access its properties, such as operation names, the coupling map connection, and initial layout, play a crucial role in shaping the behavior of the quantum circuit.

For example, with :meth:`~qiskit.providers.fake_provider.FakeQuitoV2`, you can learn about its qubit connections and the gates it uses to generate your quantum circuits.

.. testcode::

    print(f"Operation names of your backend: {backend.operation_names}")
    print(f"Coupling map connection of your backend: ",{[i for i in backend.coupling_map.get_edges()]}")

.. testoutput::

    Operation names of your backend:  ['id', 'rz', 'sx', 'x', 'cx', 'reset']
    Coupling map connection of your backend:  [(3, 4), (4, 3), (1, 3), (3, 1), (1, 2), (2, 1), (0, 1), (1, 0)]

What each optimization level does
=================================

When setting the ``optimization_level`` to 0, the resulting quantum circuit is not optimized and simply mapped to the device using a trivial layout and stochastic swap.
The coupling map, represented by the subset ``[(2,1),(1,2),(1,0),(0,1)]``, describes how the backend's physical qubits are connected.
In this configuration, the quantum circuit is transformed into a combination of one and two-qubit gates,
represented by the ``['id', 'rz', 'sx', 'x', 'cx', 'reset']``.

.. testcode::

    qc_b0 = transpile(qc,backend=backend,optimization_level=0)
    qc_b0.draw("mpl")                          

.. plot::

    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister 
    from qiskit.compiler import transpile
    from qiskit.providers.fake_provider import FakeQuitoV2

    backend = FakeQuitoV2()

    qc = QuantumCircuit(3) # Initialize the quantum circuit with 3 qubits.
    
    qc.cswap(0,1,2) # Add the cswap gate to the quantum circuit.

    qc_b0 = transpile(qc,backend=backend,optimization_level=0)
    qc_b0.draw("mpl")                          

When you set the ``optimization_level`` to 1, the circuit undergoes a light optimization process that focuses on collapsing adjacent gates 
and using heuristic algorithms to find a layout and insert swaps, with the goal to
improving the overall performance of the circuit. This results in a reduction in :class:`.CXGate` count and changes in the positions of qubits, 
following the connections ``[(2,1),(1,0),(0,1)]``. In this example, the two adjacent gates :math:`RZ(\pi/4)` and :math:`RZ(\pi/2)` are replaced with a single :math:`RZ(3\pi/4)` operation. 

.. note::
    This optimization level is the default setting.

.. testcode::

    qc_b1 = transpile(qc,backend=backend,optimization_level=1)
    qc_b1.draw("mpl")                                              

.. plot::

    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister 
    from qiskit.compiler import transpile
    from qiskit.providers.fake_provider import FakeQuitoV2

    backend = FakeQuitoV2()

    qc = QuantumCircuit(3) # Initialize the quantum circuit with 3 qubits.
    
    qc.cswap(0,1,2) # Add the cswap gate to the quantum circuit.

    qc_b1 = transpile(qc,backend=backend,optimization_level=1)
    qc_b1.draw("mpl")                                              


When you set the ``qiskit.transpile`` to 2, the circuit undergoes a medium optimization process. 
This adds gate cancellation techniques based on commutation relationships to the optimization 
techniques used in ``optimization_level`` is 1.
Additionally, the same heuristic layout and swap insertion technique is used, but with an increased number of iterations.
Depending on the circuit, this level of optimization can occasionally yield the same results as light optimization.


.. testcode::

    qc_b2 = transpile(qc,backend=backend,optimization_level=2)
    qc_b2.draw("mpl")                                                   


.. plot::

    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister 
    from qiskit.compiler import transpile
    from qiskit.providers.fake_provider import FakeQuitoV2

    backend = FakeQuitoV2()

    qc = QuantumCircuit(3) # Initialize the quantum circuit with 3 qubits.
    
    qc.cswap(0,1,2) # Add the cswap gate to the quantum circuit.

    qc_b2 = transpile(qc,backend=backend,optimization_level=2)
    qc_b2.draw("mpl")                                                   

When you set the ``optimization_level`` to 3, it enables heavy optimization. 
This level of optimization uses techniques from level 2, and also resynthesizes blocks of two-qubit gates in the circuit. 
The result of level 3 is the reduction in two-qubit gates and uses at the same inital layout as in previous levels.
Based on the operation names, results in one less :class:`.CXGate` and the addition of eight one-qubit gates.

.. testcode::

    qc_b3 = transpile(qc,backend=backend,optimization_level=3)
    qc_b3.draw("mpl")                                


.. plot::

    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister 
    from qiskit.compiler import transpile
    from qiskit.providers.fake_provider import FakeQuitoV2

    backend = FakeQuitoV2()

    qc = QuantumCircuit(3) # Initialize the quantum circuit with 3 qubits.
    
    qc.cswap(0,1,2) # Add the cswap gate to the quantum circuit.

    qc_b3 = transpile(qc,backend=backend,optimization_level=3)
    qc_b3.draw("mpl")                                


Plotting the Results
====================

You can visualize the results of your previous examples by generating a plot that show the depth, number of gates, and number of CX gates of your quantum circuits. 

.. note::
    When you set the ``optimization_level`` to 3, even if the number of one-qubit gate increases the number of two-qubit gate decreases.
    You can observe that the number of two-qubit gates (:class:`.CXGate` gates) is significantly reduced compared to other optimization levels.

.. testcode::

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    my_xticks = [str(i) for i in range(4)]
    plt.xticks(range(4), my_xticks)
    ax.plot(
        range(4),
        [qc_b0.depth(), qc_b1.depth(), qc_b2.depth(), qc_b3.depth()],
        label="Depth",
        marker="o",
        color="#6929C4",
    )
    ax.plot(
        range(4),
        [qc_b0.size(), qc_b1.size(), qc_b2.size(), qc_b3.size()],
        label="Number of gates",
        marker="o",
        color="blue",
    )
    ax.plot(
        range(4),
        [
            qc_b0.num_nonlocal_gates(),
            qc_b1.num_nonlocal_gates(),
            qc_b2.num_nonlocal_gates(),
            qc_b3.num_nonlocal_gates(),
        ],
        label="Number of two-qubit gates",
        marker="o",
        color="green",
    )

    ax.set_title("Impact of the optimization level on backend ibmq_quito")
    ax.set_xlabel("Optimization Level")
    ax.set_ylabel("Count")
    plt.legend(bbox_to_anchor=(0.75, 1.0))


.. plot::
    
    import matplotlib.pyplot as plt
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister 
    from qiskit.compiler import transpile
    from qiskit.providers.fake_provider import FakeQuitoV2
    import numpy as np

    backend = FakeQuitoV2()

    qc = QuantumCircuit(3) # Initialize the quantum circuit with 3 qubits.
    
    qc.cswap(0,1,2) # Add the cswap gate to the quantum circuit.
    
    qc0 = transpile(qc,backend=backend,optimization_level=0)
    qc1 = transpile(qc,backend=backend,optimization_level=1)
    qc2 = transpile(qc,backend=backend,optimization_level=2)
    qc3 = transpile(qc,backend=backend,optimization_level=3)


    fig, ax = plt.subplots()
    my_xticks = [str(i) for i in range(4)]
    plt.xticks(range(4), my_xticks)
    ax.plot(
        range(4),
        [qc0.depth(), qc1.depth(), qc2.depth(), qc3.depth()],
        label="Depth",
        color="#6929C4",
        marker="o",

    )
    ax.plot(
        range(4),
        [qc0.size(), qc1.size(), qc2.size(), qc3.size()],
        label="Number of gates",
        color="blue",
        marker="o",

    )
    ax.plot(
        range(4),
        [
            qc0.num_nonlocal_gates(),
            qc1.num_nonlocal_gates(),
            qc2.num_nonlocal_gates(),
            qc3.num_nonlocal_gates(),
        ],    
        label="Number of two-qubit gates",
        marker="o",

        )

    ax.set_title("Impact of the optimization level on backend ibmq_quito")
    ax.set_xlabel("Optimization Level")
    ax.set_ylabel("Count")
    plt.legend(bbox_to_anchor=(0.75, 1.0))
