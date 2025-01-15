# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

r"""
=====================================
Primitives (:mod:`qiskit.primitives`)
=====================================

.. currentmodule:: qiskit.primitives

The primitives are computational building blocks to be used in larger applications whose input
units, called primitive unified blocs (PUBs), require quantum resources to efficiently produce
outputs for.

Currently there are two types of primitives whose abstractions, in their latest versions, are
defined by :class:`~.BaseSamplerV2` and :class:`~.BaseEstimatorV2`. Samplers are responsible for
accepting quantum circuits (or sweeps of values over parameterized circuits) and sampling from their
classical output registers. Estimators accept combinations of circuits and observables (or sweeps
thereof) to estimate expectation values of the observables.

Qiskit implements a reference implementation for each of these abstractions,
:class:`~.StatevectorSampler` and :class:`~.StatevectorEstimator`.


Overview of EstimatorV2
=======================

:class:`~BaseEstimatorV2` is a primitive that estimates expectation values for provided quantum
circuit and observable combinations.

Following construction, an estimator is used by calling its :meth:`~.BaseEstimatorV2.run` method
with a list of pubs (Primitive Unified Blocs). Each pub contains three values that, together,
define a computation unit of work for the estimator to complete:

* a single :class:`~qiskit.circuit.QuantumCircuit`, possibly parametrized, whose final state we
  define as :math:`\psi(\theta)`,

* one or more observables (specified as any :class:`~.ObservablesArrayLike`, including
  :class:`~.Pauli`, :class:`~.SparsePauliOp`, ``str``) that specify which expectation values to
  estimate, denoted :math:`H_j`, and

* a collection parameter value sets to bind the circuit against, :math:`\theta_k`.

Running an estimator returns a :class:`~qiskit.primitives.BasePrimitiveJob` object, where calling
the method :meth:`~qiskit.primitives.BasePrimitiveJob.result` results in expectation value estimates
and metadata for each pub:

.. math::

    \langle\psi(\theta_k)|H_j|\psi(\theta_k)\rangle

The observables and parameter values portion of a pub can be array-valued with arbitrary dimensions,
where standard broadcasting rules are applied, so that, in turn, the estimated result for each pub
is in general array-valued as well. For more information, please check
`here <https://github.com/Qiskit/RFCs/blob/master/0015-estimator-interface.md#arrays-and
-broadcasting->`_.

Here is an example of how an estimator is used.

.. plot::
   :include-source:
   :nofigs:

    from qiskit.primitives import StatevectorEstimator as Estimator
    from qiskit.circuit.library import RealAmplitudes
    from qiskit.quantum_info import SparsePauliOp

    psi1 = RealAmplitudes(num_qubits=2, reps=2)
    psi2 = RealAmplitudes(num_qubits=2, reps=3)

    H1 = SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)])
    H2 = SparsePauliOp.from_list([("IZ", 1)])
    H3 = SparsePauliOp.from_list([("ZI", 1), ("ZZ", 1)])

    theta1 = [0, 1, 1, 2, 3, 5]
    theta2 = [0, 1, 1, 2, 3, 5, 8, 13]
    theta3 = [1, 2, 3, 4, 5, 6]

    estimator = Estimator()

    # calculate [ <psi1(theta1)|H1|psi1(theta1)> ]
    job = estimator.run([(psi1, H1, [theta1])])
    job_result = job.result() # It will block until the job finishes.
    print(f"The primitive-job finished with result {job_result}")

    # calculate [ [<psi1(theta1)|H1|psi1(theta1)>,
    #              <psi1(theta3)|H3|psi1(theta3)>],
    #             [<psi2(theta2)|H2|psi2(theta2)>] ]
    job2 = estimator.run(
        [
            (psi1, [H1, H3], [theta1, theta3]),
            (psi2, H2, theta2)
        ],
        precision=0.01
    )
    job_result = job2.result()
    print(f"The primitive-job finished with result {job_result}")


Overview of SamplerV2
=====================

:class:`~BaseSamplerV2` is a primitive that samples outputs of quantum circuits.

Following construction, a sampler is used by calling its :meth:`~.BaseSamplerV2.run` method
with a list of pubs (Primitive Unified Blocs). Each pub contains values that, together,
define a computational unit of work for the sampler to complete:

* A single :class:`~qiskit.circuit.QuantumCircuit`, possibly parameterized.

* A collection parameter value sets to bind the circuit against if it is parametric.

* Optionally, the number of shots to sample, determined in the run method if not set.

Running an estimator returns a :class:`~qiskit.primitives.BasePrimitiveJob` object, where calling
the method :meth:`~qiskit.primitives.BasePrimitiveJob.result` results in output samples and metadata
for each pub.

Here is an example of how a sampler is used.


.. plot::
   :include-source:
   :nofigs:

    from qiskit.primitives import StatevectorSampler as Sampler
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RealAmplitudes

    # create a Bell circuit
    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)
    bell.measure_all()

    # create two parameterized circuits
    pqc = RealAmplitudes(num_qubits=2, reps=2)
    pqc.measure_all()
    pqc2 = RealAmplitudes(num_qubits=2, reps=3)
    pqc2.measure_all()

    theta1 = [0, 1, 1, 2, 3, 5]
    theta2 = [0, 1, 2, 3, 4, 5, 6, 7]

    # initialization of the sampler
    sampler = Sampler()

    # collect 128 shots from the Bell circuit
    job = sampler.run([bell], shots=128)
    job_result = job.result()
    print(f"The primitive-job finished with result {job_result}")

    # run a sampler job on the parameterized circuits
    job2 = sampler.run([(pqc, theta1), (pqc2, theta2)])
    job_result = job2.result()
    print(f"The primitive-job finished with result {job_result}")


Overview of EstimatorV1
=======================

Estimator class estimates expectation values of quantum circuits and observables.

An estimator is initialized with an empty parameter set. The estimator is used to
create a :class:`~qiskit.providers.JobV1`, via the
:meth:`qiskit.primitives.Estimator.run()` method. This method is called
with the following parameters

* quantum circuits (:math:`\psi_i(\theta)`): list of (parameterized) quantum circuits
  (a list of :class:`~qiskit.circuit.QuantumCircuit` objects).

* observables (:math:`H_j`): a list of :class:`~qiskit.quantum_info.SparsePauliOp`
  objects.

* parameter values (:math:`\theta_k`): list of sets of values
  to be bound to the parameters of the quantum circuits
  (list of list of float).

The method returns a :class:`~qiskit.providers.JobV1` object, calling
:meth:`qiskit.providers.JobV1.result()` yields the
a list of expectation values plus optional metadata like confidence intervals for
the estimation.

.. math::

    \langle\psi_i(\theta_k)|H_j|\psi_i(\theta_k)\rangle

Here is an example of how the estimator is used.

.. plot::
   :include-source:
   :nofigs:

    from qiskit.primitives import Estimator
    from qiskit.circuit.library import RealAmplitudes
    from qiskit.quantum_info import SparsePauliOp

    psi1 = RealAmplitudes(num_qubits=2, reps=2)
    psi2 = RealAmplitudes(num_qubits=2, reps=3)

    H1 = SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)])
    H2 = SparsePauliOp.from_list([("IZ", 1)])
    H3 = SparsePauliOp.from_list([("ZI", 1), ("ZZ", 1)])

    theta1 = [0, 1, 1, 2, 3, 5]
    theta2 = [0, 1, 1, 2, 3, 5, 8, 13]
    theta3 = [1, 2, 3, 4, 5, 6]

    estimator = Estimator()

    # calculate [ <psi1(theta1)|H1|psi1(theta1)> ]
    job = estimator.run([psi1], [H1], [theta1])
    job_result = job.result() # It will block until the job finishes.
    print(f"The primitive-job finished with result {job_result}")

    # calculate [ <psi1(theta1)|H1|psi1(theta1)>,
    #             <psi2(theta2)|H2|psi2(theta2)>,
    #             <psi1(theta3)|H3|psi1(theta3)> ]
    job2 = estimator.run(
        [psi1, psi2, psi1],
        [H1, H2, H3],
        [theta1, theta2, theta3]
    )
    job_result = job2.result()
    print(f"The primitive-job finished with result {job_result}")


Overview of SamplerV1
=====================

Sampler class calculates probabilities or quasi-probabilities of bitstrings from quantum circuits.

A sampler is initialized with an empty parameter set. The sampler is used to
create a :class:`~qiskit.providers.JobV1`, via the :meth:`qiskit.primitives.Sampler.run()`
method. This method is called with the following parameters

* quantum circuits (:math:`\psi_i(\theta)`): list of (parameterized) quantum circuits.
  (a list of :class:`~qiskit.circuit.QuantumCircuit` objects)

* parameter values (:math:`\theta_k`): list of sets of parameter values
  to be bound to the parameters of the quantum circuits.
  (list of list of float)

The method returns a :class:`~qiskit.providers.JobV1` object, calling
:meth:`qiskit.providers.JobV1.result()` yields a :class:`~qiskit.primitives.SamplerResult`
object, which contains probabilities or quasi-probabilities of bitstrings,
plus optional metadata like error bars in the samples.

Here is an example of how sampler is used.

.. plot::
   :include-source:
   :nofigs:

    from qiskit.primitives import Sampler
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RealAmplitudes

    # a Bell circuit
    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)
    bell.measure_all()

    # two parameterized circuits
    pqc = RealAmplitudes(num_qubits=2, reps=2)
    pqc.measure_all()
    pqc2 = RealAmplitudes(num_qubits=2, reps=3)
    pqc2.measure_all()

    theta1 = [0, 1, 1, 2, 3, 5]
    theta2 = [0, 1, 2, 3, 4, 5, 6, 7]

    # initialization of the sampler
    sampler = Sampler()

    # Sampler runs a job on the Bell circuit
    job = sampler.run(
        circuits=[bell], parameter_values=[[]], parameters=[[]]
    )
    job_result = job.result()
    print([q.binary_probabilities() for q in job_result.quasi_dists])

    # Sampler runs a job on the parameterized circuits
    job2 = sampler.run(
        circuits=[pqc, pqc2],
        parameter_values=[theta1, theta2],
        parameters=[pqc.parameters, pqc2.parameters])
    job_result = job2.result()
    print([q.binary_probabilities() for q in job_result.quasi_dists])


Migration from Primitives V1 to V2
==================================

The formal distinction between the Primitives V1 and V2 APIs are the base classes from which
primitives implementations inherit, which are all listed at the bottom of the page. At a conceptual
level, however, here are some notable differences keep in mind when migrating from V1 to V2:

1. The V2 primitives favour vectorized inputs, where single circuits can be grouped with
   vector-valued (or more generally, array-valued) specifications. Each group is called a
   primitive unified bloc (pub), and each pub gets its own result. For example, in the estimator,
   you can compare the following differences:

   .. code-block:: python

      # Favoured V2 pattern. There is only one pub here, but there could be more.
      job = estimator_v2.run([(circuit, [obs1, obs2, obs3, obs4])])
      evs = job.result()[0].data.evs

      # V1 equivalent, where the same circuit must be provided four times.
      job = estimator_v1.run([circuit] * 4, [obs1, obs2, obs3, obs4])
      evs = job.result().values

   Not shown in the above example, for brevity, is that the circuit can be parametric, with arrays
   of parameter value sets broadcasted against the array of observables. The sampler is similar,
   but with no observables:

   .. code-block:: python

      # Favoured V2 pattern. There is only one pub here, but there could be more.
      job = sampler_v2.run([(circuit, [vals1, vals2, vals3])])
      samples = job.result()[0].data

      # V1 equivalent, where the same circuit must be provided three times.
      sampler_v1.run([circuit] * 3, [vals1, vals2, vals3])
      quasi_dists = job.result().quasi_dists

2. The V2 sampler returns samples of classical outcomes, preserving the shot order in which they
   were measured. This is in contrast to the V1 sampler that outputs quasi-probability distributions
   which are instead an *estimate of the distribution* over classical outcomes. Moreover, the V2
   sampler result objects organize data in terms of their input circuits' classical register
   names, which provides natural compatibility with dynamic circuits.

   The closest analog of quasi-probability distributions in the V2 interface is the
   :meth:`~.BitArray.get_counts` method, shown in the example below. However, we emphasize that
   for utility scale experiments (100+ qubits), the chances of measuring the same bitstring twice
   are small, so that binning like counts in a dictionary format will not typically be an efficient
   data processing strategy.

   .. code-block:: python

      circuit = QuantumCircuit(QuantumRegister(2, "qreg"), ClassicalRegister(2, "alpha"))
      circuit.h(0)
      circuit.cx(0, 1)
      circuit.measure([0, 1], [0, 1])

      # V1 sampler usage
      result = sampler_v1.run([circuit]).result()
      quasi_dist = result.quasi_dists[0]

      # V2 sampler usage
      result = sampler_v2.run([circuit]).result()
      # these are the bit values from the alpha register, over all shots
      bitvals = result[0].data.alpha
      # we can use it to generate a Counts mapping, which is similar to a quasi prob distribution
      counts = bitvals.get_counts()
      # which can in turn be converted to the V1 type through normalization
      quasi_dist = QuasiDistribution({outcome: freq / shots for outcome, freq in counts.items()})

3. The V2 primitives have brought the concept of sampling overhead, inherent to all quantum systems
   via their inherent probabilistic nature, out of the options and into the API itself. For the
   sampler, this means that the ``shots`` argument is now part of the :meth:`~.BaseSamplerV2.run`
   signature, and moreover that each pub is able to specify its own value for ``shots``, which takes
   precedence over any value given to the method. The sampler has an analogous ``precision``
   argument that specifies the error bars that the primitive implementation should target for
   expectation values estimates.

   This concept is not present in the API of the V1 primitives, though all implementations of the
   V1 primitives have related settings somewhere in their options.

   .. code-block:: python

      # Sample two circuits at 128 shots each.
      sampler_v2.run([circuit1, circuit2], shots=128)

      # Sample two circuits at different amounts of shots. The "None"s are necessary as placeholders
      # for the lack of parameter values in this example.
      sampler_v2.run([(circuit1, None, 123), (circuit2, None, 456)])

      # Estimate expectation values for two pubs, both with 0.05 precision.
      estimator_v2.run([(circuit1, obs_array1), (circuit2, obs_array_2)], precision=0.05)



Primitives API
==============

Estimator V2
------------

.. autosummary::
   :toctree: ../stubs/

   BaseEstimatorV2
   StatevectorEstimator
   BackendEstimatorV2

Sampler V2
----------

.. autosummary::
   :toctree: ../stubs/

   BaseSamplerV2
   StatevectorSampler
   BackendSamplerV2

Results V2
----------

.. autosummary::
   :toctree: ../stubs/

   BitArray
   DataBin
   PrimitiveResult
   PubResult
   SamplerPubResult
   BasePrimitiveJob
   PrimitiveJob

Estimator V1
------------

.. autosummary::
   :toctree: ../stubs/

   BaseEstimator
   BaseEstimatorV1
   Estimator
   BackendEstimator
   EstimatorResult


Sampler V1
----------

.. autosummary::
   :toctree: ../stubs/

   BaseSampler
   BaseSamplerV1
   Sampler
   BackendSampler
   SamplerResult


Containers
----------

.. autosummary::
   :toctree: ../stubs/

  BitArray
  DataBin
  PrimitiveResult
  PubResult
  EstimatorPubLike
  SamplerPubLike
  SamplerPubResult
  BindingsArrayLike
  ObservableLike
  ObservablesArrayLike

"""

from .backend_estimator import BackendEstimator
from .backend_sampler import BackendSampler
from .base import (
    BaseEstimator,
    BaseEstimatorV1,
    BaseEstimatorV2,
    BaseSampler,
    BaseSamplerV1,
    BaseSamplerV2,
)
from .base.estimator_result import EstimatorResult
from .base.sampler_result import SamplerResult
from .containers import (
    BitArray,
    DataBin,
    PrimitiveResult,
    PubResult,
    EstimatorPubLike,
    SamplerPubLike,
    SamplerPubResult,
    BindingsArrayLike,
    ObservableLike,
    ObservablesArrayLike,
)
from .estimator import Estimator
from .primitive_job import BasePrimitiveJob, PrimitiveJob
from .sampler import Sampler
from .statevector_estimator import StatevectorEstimator
from .statevector_sampler import StatevectorSampler
from .backend_estimator_v2 import BackendEstimatorV2
from .backend_sampler_v2 import BackendSamplerV2
