# Qiskit

[![License](https://img.shields.io/github/license/Qiskit/qiskit.svg?)](https://opensource.org/licenses/Apache-2.0) <!--- long-description-skip-begin -->
[![Current Release](https://img.shields.io/github/release/Qiskit/qiskit.svg?logo=Qiskit)](https://github.com/Qiskit/qiskit/releases)
[![Extended Support Release](https://img.shields.io/github/v/release/Qiskit/qiskit?sort=semver&filter=0.*&logo=Qiskit&label=extended%20support)](https://github.com/Qiskit/qiskit/releases?q=tag%3A0)
[![Downloads](https://img.shields.io/pypi/dm/qiskit.svg)](https://pypi.org/project/qiskit/)
[![Coverage Status](https://coveralls.io/repos/github/Qiskit/qiskit/badge.svg?branch=main)](https://coveralls.io/github/Qiskit/qiskit?branch=main)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/qiskit)
[![Minimum rustc 1.70](https://img.shields.io/badge/rustc-1.70+-blue.svg)](https://rust-lang.github.io/rfcs/2495-min-rust-version.html)
[![Downloads](https://static.pepy.tech/badge/qiskit)](https://pepy.tech/project/qiskit)<!--- long-description-skip-end -->
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2583252.svg)](https://doi.org/10.5281/zenodo.2583252)

**Qiskit**  is an open-source SDK for working with quantum computers at the level of extended quantum circuits, operators, and primitives.

This library is the core component of Qiskit, which contains the building blocks for creating and working with quantum circuits, quantum operators, and primitive functions (Sampler and Estimator).
It also contains a transpiler that supports optimizing quantum circuits, and a quantum information toolbox for creating advanced operators.

For more details on how to use Qiskit, refer to the documentation located here:

<https://docs.quantum.ibm.com/>


## Installation

> [!WARNING]
> Do not try to upgrade an existing Qiskit 0.* environment to Qiskit 1.0 in-place. [Read more](https://docs.quantum.ibm.com/migration-guides/qiskit-1.0-installation).

We encourage installing Qiskit via ``pip``:

```bash
pip install qiskit
```

Pip will handle all dependencies automatically and you will always install the latest (and well-tested) version.

To install from source, follow the instructions in the [documentation](https://docs.quantum.ibm.com/guides/install-qiskit-source).

## Create your first quantum program in Qiskit

Now that Qiskit is installed, it's time to begin working with Qiskit. The essential parts of a quantum program are:
1. Define and build a quantum circuit that represents the quantum state
2. Define the classical output by measurements or a set of observable operators
3. Depending on the output, use the Sampler primitive to sample outcomes or the Estimator primitive to estimate expectation values.

Create an example quantum circuit using the `QuantumCircuit` class:

```python
import numpy as np
from qiskit import QuantumCircuit

# 1. A quantum circuit for preparing the quantum state |000> + i |111> / √2
qc = QuantumCircuit(3)
qc.h(0)             # generate superposition
qc.p(np.pi / 2, 0)  # add quantum phase
qc.cx(0, 1)         # 0th-qubit-Controlled-NOT gate on 1st qubit
qc.cx(0, 2)         # 0th-qubit-Controlled-NOT gate on 2nd qubit
```

This simple example creates an entangled state known as a [GHZ state](https://en.wikipedia.org/wiki/Greenberger%E2%80%93Horne%E2%80%93Zeilinger_state) $(|000\rangle + i|111\rangle)/\sqrt{2}$. It uses the standard quantum gates: Hadamard gate (`h`), Phase gate (`p`), and CNOT gate (`cx`). 

Once you've made your first quantum circuit, choose which primitive you will use. Starting with the Sampler,
we use `measure_all(inplace=False)` to get a copy of the circuit in which all the qubits are measured:

```python
# 2. Add the classical output in the form of measurement of all qubits
qc_measured = qc.measure_all(inplace=False)

# 3. Execute using the Sampler primitive
from qiskit.primitives import StatevectorSampler
sampler = StatevectorSampler()
job = sampler.run([qc_measured], shots=1000)
result = job.result()
print(f" > Counts: {result[0].data["meas"].get_counts()}")
```
Running this will give an outcome similar to `{'000': 497, '111': 503}` which is `000` 50% of the time and `111` 50% of the time up to statistical fluctuations.
To illustrate the power of the Estimator, we now use the quantum information toolbox to create the operator $XXY+XYX+YXX-YYY$ and pass it to the `run()` function, along with our quantum circuit. Note that the Estimator requires a circuit _**without**_ measurements, so we use the `qc` circuit we created earlier.

```python
# 2. Define the observable to be measured 
from qiskit.quantum_info import SparsePauliOp
operator = SparsePauliOp.from_list([("XXY", 1), ("XYX", 1), ("YXX", 1), ("YYY", -1)])

# 3. Execute using the Estimator primitive
from qiskit.primitives import StatevectorEstimator
estimator = StatevectorEstimator()
job = estimator.run([(qc, operator)], precision=1e-3)
result = job.result()
print(f" > Expectation values: {result[0].data.evs}")
```

Running this will give the outcome `4`. For fun, try to assign a value of +/- 1 to each single-qubit operator X and Y 
and see if you can achieve this outcome. (Spoiler alert: this is not possible!)

Using the Qiskit-provided `qiskit.primitives.StatevectorSampler` and `qiskit.primitives.StatevectorEstimator` will not take you very far.
The power of quantum computing cannot be simulated on classical computers and you need to use real quantum hardware to scale to larger quantum circuits.
However, running a quantum circuit on hardware requires rewriting to the basis gates and connectivity of the quantum hardware.
The tool that does this is the [transpiler](https://docs.quantum.ibm.com/api/qiskit/transpiler), and Qiskit includes transpiler passes for synthesis, optimization, mapping, and scheduling.
However, it also includes a default compiler, which works very well in most examples.
The following code will map the example circuit to the `basis_gates = ["cz", "sx", "rz"]` and a linear chain of qubits $0 \rightarrow 1 \rightarrow 2$ with the `coupling_map = [[0, 1], [1, 2]]`.

```python
from qiskit import transpile
qc_transpiled = transpile(qc, basis_gates=["cz", "sx", "rz"], coupling_map=[[0, 1], [1, 2]], optimization_level=3)
```

### Executing your code on real quantum hardware

Qiskit provides an abstraction layer that lets users run quantum circuits on hardware from any vendor that provides a compatible interface. 
The best way to use Qiskit is with a runtime environment that provides optimized implementations of Sampler and Estimator for a given hardware platform. This runtime may involve using pre- and post-processing, such as optimized transpiler passes with error suppression, error mitigation, and, eventually, error correction built in. A runtime implements `qiskit.primitives.BaseSamplerV2` and `qiskit.primitives.BaseEstimatorV2` interfaces. For example,
some packages that provide implementations of a runtime primitive implementation are:

* https://github.com/Qiskit/qiskit-ibm-runtime

Qiskit also provides a lower-level abstract interface for describing quantum backends. This interface, located in
``qiskit.providers``, defines an abstract `BackendV2` class that providers can implement to represent their
hardware or simulators to Qiskit. The backend class includes a common interface for executing circuits on the backends; however, in this interface each provider may perform different types of pre- and post-processing and return outcomes that are vendor-defined. Some examples of published provider packages that interface with real hardware are:

* https://github.com/qiskit-community/qiskit-ionq
* https://github.com/qiskit-community/qiskit-aqt-provider
* https://github.com/qiskit-community/qiskit-braket-provider
* https://github.com/qiskit-community/qiskit-quantinuum-provider
* https://github.com/rigetti/qiskit-rigetti

<!-- This is not an exhaustive list, and if you maintain a provider package please feel free to open a PR to add new providers -->

You can refer to the documentation of these packages for further instructions
on how to get access and use these systems.

## Contribution Guidelines

If you'd like to contribute to Qiskit, please take a look at our
[contribution guidelines](CONTRIBUTING.md). By participating, you are expected to uphold our [code of conduct](CODE_OF_CONDUCT.md).

We use [GitHub issues](https://github.com/Qiskit/qiskit/issues) for tracking requests and bugs. Please
[join the Qiskit Slack community](https://qisk.it/join-slack) for discussion, comments, and questions.
For questions related to running or using Qiskit, [Stack Overflow has a `qiskit`](https://stackoverflow.com/questions/tagged/qiskit).
For questions on quantum computing with Qiskit, use the `qiskit` tag in the [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com/questions/tagged/qiskit) (please, read first the [guidelines on how to ask](https://quantumcomputing.stackexchange.com/help/how-to-ask) in that forum).


## Authors and Citation

Qiskit is the work of [many people](https://github.com/Qiskit/qiskit/graphs/contributors) who contribute
to the project at different levels. If you use Qiskit, please cite as per the included [BibTeX file](CITATION.bib).

## Changelog and Release Notes

The changelog for a particular release is dynamically generated and gets
written to the release page on Github for each release. For example, you can
find the page for the `0.46.0` release here:

<https://github.com/Qiskit/qiskit/releases/tag/0.46.0>

The changelog for the current release can be found in the releases tab:
[![Releases](https://img.shields.io/github/release/Qiskit/qiskit.svg?style=flat&label=)](https://github.com/Qiskit/qiskit/releases)
The changelog provides a quick overview of notable changes for a given
release.

Additionally, as part of each release, detailed release notes are written to
document in detail what has changed as part of a release. This includes any
documentation on potential breaking changes on upgrade and new features. See [all release notes here](https://docs.quantum.ibm.com/api/qiskit/release-notes).

## Acknowledgements

We acknowledge partial support for Qiskit development from the DOE Office of Science National Quantum Information Science Research Centers, Co-design Center for Quantum Advantage (C2QA) under contract number DE-SC0012704.

## License

[Apache License 2.0](LICENSE.txt)
