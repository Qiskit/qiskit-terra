# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""Backend abstract interface for providers."""


from abc import ABC
from abc import abstractmethod
import datetime
from typing import List, Union, Iterable, Tuple

from qiskit.providers.provider import Provider
from qiskit.providers.models.backendstatus import BackendStatus
from qiskit.circuit.gate import Instruction
from qiskit.utils import deprecate_func
from qiskit.utils.deprecate_pulse import deprecate_pulse_dependency


class Backend:
    """Base common type for all versioned Backend abstract classes.

    Note this class should not be inherited from directly, it is intended
    to be used for type checking. When implementing a provider you should use
    the versioned abstract classes as the parent class and not this class
    directly.
    """

    version = 0


class BackendV1(Backend, ABC):
    """Abstract class for Backends

    This abstract class is to be used for Backend objects.
    There are several classes of information contained in a Backend.
    The first are the attributes of the class itself. These should be used to
    define the immutable characteristics of the backend. The ``options``
    attribute of the backend is used to contain the dynamic user configurable
    options of the backend. It should be used more for runtime options
    that configure how the backend is used. For example, something like a
    ``shots`` field for a backend that runs experiments which would contain an
    int for how many shots to execute. The ``properties`` attribute is
    optionally defined :class:`~qiskit.providers.models.BackendProperties`
    object and is used to return measured properties, or properties
    of a backend that may change over time. The simplest example of this would
    be a version string, which will change as a backend is updated, but also
    could be something like noise parameters for backends that run experiments.

    This first version of the Backend abstract class is written to be mostly
    backwards compatible with the legacy providers interface. This includes reusing
    the model objects :class:`~qiskit.providers.models.BackendProperties` and
    :class:`~qiskit.providers.models.BackendConfiguration`. This was done to
    ease the transition for users and provider maintainers to the new versioned providers.
    Expect, future versions of this abstract class to change the data model and
    interface.

    Subclasses of this should override the public method :meth:`run` and the internal
    :meth:`_default_options`:

    .. automethod:: _default_options
    """

    version = 1

    @deprecate_func(
        since="1.2",
        removal_timeline="in the 2.0 release",
        additional_msg="If the backend only encapsulates a hardware description, "
        "consider constructing a Target directly. If it is part of a provider "
        "that gives access to execution, consider using Primitives instead. "
        "Alternatively, consider moving to BackendV2 (see https://qisk.it/backendV1-to-V2).",
    )
    def __init__(self, configuration, provider=None, **fields):
        """Initialize a backend class

        Args:
            configuration (BackendConfiguration): A backend configuration
                object for the backend object.
            provider (qiskit.providers.Provider): Optionally, the provider
                object that this Backend comes from.
            fields: kwargs for the values to use to override the default
                options.
        Raises:
            AttributeError: if input field not a valid options

        ..
            This next bit is necessary just because autosummary generally won't summarise private
            methods; changing that behavior would have annoying knock-on effects through all the
            rest of the documentation, so instead we just hard-code the automethod directive.
        """
        self._configuration = configuration
        self._options = self._default_options()
        self._provider = provider
        if fields:
            for field in fields:
                if field not in self._options.data:
                    raise AttributeError(f"Options field {field} is not valid for this backend")
            self._options.update_config(**fields)

    @classmethod
    @abstractmethod
    def _default_options(cls):
        """Return the default options

        This method will return a :class:`qiskit.providers.Options`
        subclass object that will be used for the default options. These
        should be the default parameters to use for the options of the
        backend.

        Returns:
            qiskit.providers.Options: A options object with
                default values set
        """

    def set_options(self, **fields):
        """Set the options fields for the backend

        This method is used to update the options of a backend. If
        you need to change any of the options prior to running just
        pass in the kwarg with the new value for the options.

        Args:
            fields: The fields to update the options

        Raises:
            AttributeError: If the field passed in is not part of the
                options
        """
        for field in fields:
            if not hasattr(self._options, field):
                raise AttributeError(f"Options field {field} is not valid for this backend")
        self._options.update_options(**fields)

    def configuration(self):
        """Return the backend configuration.

        Returns:
            BackendConfiguration: the configuration for the backend.
        """
        return self._configuration

    def properties(self):
        """Return the backend properties.

        Returns:
            BackendProperties: the configuration for the backend. If the backend
            does not support properties, it returns ``None``.
        """
        return None

    def provider(self):
        """Return the backend Provider.

        Returns:
            Provider: the Provider responsible for the backend.
        """
        return self._provider

    def status(self):
        """Return the backend status.

        Returns:
            BackendStatus: the status of the backend.
        """
        return BackendStatus(
            backend_name=self.name(),
            backend_version="1",
            operational=True,
            pending_jobs=0,
            status_msg="",
        )

    def name(self):
        """Return the backend name.

        Returns:
            str: the name of the backend.
        """
        return self._configuration.backend_name

    def __str__(self):
        return self.name()

    def __repr__(self):
        """Official string representation of a Backend.

        Note that, by Qiskit convention, it is consciously *not* a fully valid
        Python expression. Subclasses should provide 'a string of the form
        <...some useful description...>'. [0]

        [0] https://docs.python.org/3/reference/datamodel.html#object.__repr__
        """
        return f"<{self.__class__.__name__}('{self.name()}')>"

    @property
    def options(self):
        """Return the options for the backend

        The options of a backend are the dynamic parameters defining
        how the backend is used. These are used to control the :meth:`run`
        method.
        """
        return self._options

    @abstractmethod
    def run(self, run_input, **options):
        """Run on the backend.

        This method returns a :class:`~qiskit.providers.Job` object
        that runs circuits. Depending on the backend this may be either an async
        or sync call. It is at the discretion of the provider to decide whether
        running should block until the execution is finished or not: the Job
        class can handle either situation.

        Args:
            run_input (QuantumCircuit or Schedule or list): An individual or a
                list of :class:`~qiskit.circuit.QuantumCircuit` or
                :class:`~qiskit.pulse.Schedule` objects to run on the backend.
                For legacy providers migrating to the new versioned providers,
                provider interface a :class:`~qiskit.qobj.QasmQobj` or
                :class:`~qiskit.qobj.PulseQobj` objects should probably be
                supported too (but deprecated) for backwards compatibility. Be
                sure to update the docstrings of subclasses implementing this
                method to document that. New provider implementations should not
                do this though as :mod:`qiskit.qobj` will be deprecated and
                removed along with the legacy providers interface.
            options: Any kwarg options to pass to the backend for running the
                config. If a key is also present in the options
                attribute/object then the expectation is that the value
                specified will be used instead of what's set in the options
                object.
        Returns:
            Job: The job object for the run
        """
        pass


class QubitProperties:
    """A representation of the properties of a qubit on a backend.

    This class provides the optional properties that a backend can provide for
    a qubit. These represent the set of qubit properties that Qiskit can
    currently work with if present. However if your backend provides additional
    properties of qubits you should subclass this to add additional custom
    attributes for those custom/additional properties provided by the backend.
    """

    __slots__ = ("t1", "t2", "frequency")

    def __init__(self, t1=None, t2=None, frequency=None):
        """Create a new :class:`QubitProperties` object.

        Args:
            t1: The T1 time for a qubit in seconds
            t2: The T2 time for a qubit in seconds
            frequency: The frequency of a qubit in Hz
        """
        self.t1 = t1
        self.t2 = t2
        self.frequency = frequency

    def __repr__(self):
        return f"QubitProperties(t1={self.t1}, t2={self.t2}, " f"frequency={self.frequency})"


class BackendV2(Backend, ABC):
    """Abstract class for Backends

    This abstract class is to be used for all Backend objects created by a
    provider. This version differs from earlier abstract Backend classes in
    that the configuration attribute no longer exists. Instead, attributes
    exposing equivalent required immutable properties of the backend device
    are added. For example ``backend.configuration().n_qubits`` is accessible
    from ``backend.num_qubits`` now.

    The ``options`` attribute of the backend is used to contain the dynamic
    user configurable options of the backend. It should be used more for
    runtime options that configure how the backend is used. For example,
    something like a ``shots`` field for a backend that runs experiments which
    would contain an int for how many shots to execute.

    If migrating a provider from :class:`~qiskit.providers.BackendV1`
    one thing to keep in mind is for
    backwards compatibility you might need to add a configuration method that
    will build a :class:`~qiskit.providers.models.BackendConfiguration` object
    and :class:`~qiskit.providers.models.BackendProperties` from the attributes
    defined in this class for backwards compatibility.

    A backend object can optionally contain methods named
    ``get_translation_stage_plugin`` and ``get_scheduling_stage_plugin``. If these
    methods are present on a backend object and this object is used for
    :func:`~.transpile` or :func:`~.generate_preset_pass_manager` the
    transpilation process will default to using the output from those methods
    as the scheduling stage and the translation compilation stage. This
    enables a backend which has custom requirements for compilation to specify a
    stage plugin for these stages to enable custom transformation of
    the circuit to ensure it is runnable on the backend. These hooks are enabled
    by default and should only be used to enable extra compilation steps
    if they are **required** to ensure a circuit is executable on the backend or
    have the expected level of performance. These methods are passed no input
    arguments and are expected to return a ``str`` representing the method name
    which should be a stage plugin (see: :mod:`qiskit.transpiler.preset_passmanagers.plugin`
    for more details on plugins). The typical expected use case is for a backend
    provider to implement a stage plugin for ``translation`` or ``scheduling``
    that contains the custom compilation passes and then for the hook methods on
    the backend object to return the plugin name so that :func:`~.transpile` will
    use it by default when targetting the backend.

    Subclasses of this should override the public method :meth:`run` and the internal
    :meth:`_default_options`:

    .. automethod:: _default_options
    """

    version = 2

    def __init__(
        self,
        provider: Provider = None,
        name: str = None,
        description: str = None,
        online_date: datetime.datetime = None,
        backend_version: str = None,
        **fields,
    ):
        """Initialize a BackendV2 based backend

        Args:
            provider: An optional backwards reference to the
                :class:`~qiskit.providers.Provider` object that the backend
                is from
            name: An optional name for the backend
            description: An optional description of the backend
            online_date: An optional datetime the backend was brought online
            backend_version: An optional backend version string. This differs
                from the :attr:`~qiskit.providers.BackendV2.version` attribute
                as :attr:`~qiskit.providers.BackendV2.version` is for the
                abstract :attr:`~qiskit.providers.Backend` abstract interface
                version of the object while ``backend_version`` is for
                versioning the backend itself.
            fields: kwargs for the values to use to override the default
                options.

        Raises:
            AttributeError: If a field is specified that's outside the backend's
                options
        """

        self._options = self._default_options()
        self._provider = provider
        if fields:
            for field in fields:
                if field not in self._options.data:
                    raise AttributeError(f"Options field {field} is not valid for this backend")
            self._options.update_config(**fields)
        self.name = name
        """Name of the backend."""
        self.description = description
        """Optional human-readable description."""
        self.online_date = online_date
        """Date that the backend came online."""
        self.backend_version = backend_version
        """Version of the backend being provided.  This is not the same as
        :attr:`.BackendV2.version`, which is the version of the :class:`~.providers.Backend`
        abstract interface."""
        self._coupling_map = None

    @property
    def instructions(self) -> List[Tuple[Instruction, Tuple[int]]]:
        """A list of Instruction tuples on the backend of the form ``(instruction, (qubits)``"""
        return self.target.instructions

    @property
    def operations(self) -> List[Instruction]:
        """A list of :class:`~qiskit.circuit.Instruction` instances that the backend supports."""
        return list(self.target.operations)

    @property
    def operation_names(self) -> List[str]:
        """A list of instruction names that the backend supports."""
        return list(self.target.operation_names)

    @property
    @abstractmethod
    def target(self):
        """A :class:`qiskit.transpiler.Target` object for the backend.

        :rtype: Target
        """
        pass

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits the backend has."""
        return self.target.num_qubits

    @property
    def coupling_map(self):
        """Return the :class:`~qiskit.transpiler.CouplingMap` object"""
        if self._coupling_map is None:
            self._coupling_map = self.target.build_coupling_map()
        return self._coupling_map

    @property
    def instruction_durations(self):
        """Return the :class:`~qiskit.transpiler.InstructionDurations` object."""
        return self.target.durations()

    @property
    @abstractmethod
    def max_circuits(self):
        """The maximum number of circuits (or Pulse schedules) that can be
        run in a single job.

        If there is no limit this will return None
        """
        pass

    @classmethod
    @abstractmethod
    def _default_options(cls):
        """Return the default options

        This method will return a :class:`qiskit.providers.Options`
        subclass object that will be used for the default options. These
        should be the default parameters to use for the options of the
        backend.

        Returns:
            qiskit.providers.Options: A options object with
                default values set
        """
        pass

    @property
    def dt(self) -> Union[float, None]:
        """Return the system time resolution of input signals

        This is required to be implemented if the backend supports Pulse
        scheduling.

        Returns:
            The input signal timestep in seconds. If the backend doesn't define ``dt``, ``None`` will
            be returned.
        """
        return self.target.dt

    @property
    def dtm(self) -> float:
        """Return the system time resolution of output signals

        Returns:
            The output signal timestep in seconds.

        Raises:
            NotImplementedError: if the backend doesn't support querying the
                output signal timestep
        """
        raise NotImplementedError

    @property
    def meas_map(self) -> List[List[int]]:
        """Return the grouping of measurements which are multiplexed

        This is required to be implemented if the backend supports Pulse
        scheduling.

        Returns:
            The grouping of measurements which are multiplexed

        Raises:
            NotImplementedError: if the backend doesn't support querying the
                measurement mapping
        """
        raise NotImplementedError

    @property
    @deprecate_pulse_dependency(is_property=True)
    def instruction_schedule_map(self):
        """Return the :class:`~qiskit.pulse.InstructionScheduleMap` for the
        instructions defined in this backend's target."""
        return self._instruction_schedule_map

    @property
    def _instruction_schedule_map(self):
        """An alternative private path to be used internally to avoid pulse deprecation warnings."""
        return self.target._get_instruction_schedule_map()

    def qubit_properties(
        self, qubit: Union[int, List[int]]
    ) -> Union[QubitProperties, List[QubitProperties]]:
        """Return QubitProperties for a given qubit.

        If there are no defined or the backend doesn't support querying these
        details this method does not need to be implemented.

        Args:
            qubit: The qubit to get the
                :class:`.QubitProperties` object for. This can
                be a single integer for 1 qubit or a list of qubits and a list
                of :class:`.QubitProperties` objects will be
                returned in the same order
        Returns:
            The :class:`~.QubitProperties` object for the
            specified qubit. If a list of qubits is provided a list will be
            returned. If properties are missing for a qubit this can be
            ``None``.

        Raises:
            NotImplementedError: if the backend doesn't support querying the
                qubit properties
        """
        # Since the target didn't always have a qubit properties attribute
        # to ensure the behavior here is backwards compatible with earlier
        # BacekendV2 implementations where this would raise a NotImplemented
        # error.
        if self.target.qubit_properties is None:
            raise NotImplementedError
        if isinstance(qubit, int):
            return self.target.qubit_properties[qubit]
        return [self.target.qubit_properties[q] for q in qubit]

    @deprecate_pulse_dependency
    def drive_channel(self, qubit: int):
        """Return the drive channel for the given qubit.

        This is required to be implemented if the backend supports Pulse
        scheduling.

        Returns:
            DriveChannel: The Qubit drive channel

        Raises:
            NotImplementedError: if the backend doesn't support querying the
                measurement mapping
        """
        raise NotImplementedError

    @deprecate_pulse_dependency
    def measure_channel(self, qubit: int):
        """Return the measure stimulus channel for the given qubit.

        This is required to be implemented if the backend supports Pulse
        scheduling.

        Returns:
            MeasureChannel: The Qubit measurement stimulus line

        Raises:
            NotImplementedError: if the backend doesn't support querying the
                measurement mapping
        """
        raise NotImplementedError

    @deprecate_pulse_dependency
    def acquire_channel(self, qubit: int):
        """Return the acquisition channel for the given qubit.

        This is required to be implemented if the backend supports Pulse
        scheduling.

        Returns:
            AcquireChannel: The Qubit measurement acquisition line.

        Raises:
            NotImplementedError: if the backend doesn't support querying the
                measurement mapping
        """
        raise NotImplementedError

    @deprecate_pulse_dependency
    def control_channel(self, qubits: Iterable[int]):
        """Return the secondary drive channel for the given qubit

        This is typically utilized for controlling multiqubit interactions.
        This channel is derived from other channels.

        This is required to be implemented if the backend supports Pulse
        scheduling.

        Args:
            qubits: Tuple or list of qubits of the form
                ``(control_qubit, target_qubit)``.

        Returns:
            List[ControlChannel]: The multi qubit control line.

        Raises:
            NotImplementedError: if the backend doesn't support querying the
                measurement mapping
        """
        raise NotImplementedError

    def set_options(self, **fields):
        """Set the options fields for the backend

        This method is used to update the options of a backend. If
        you need to change any of the options prior to running just
        pass in the kwarg with the new value for the options.

        Args:
            fields: The fields to update the options

        Raises:
            AttributeError: If the field passed in is not part of the
                options
        """
        for field in fields:
            if not hasattr(self._options, field):
                raise AttributeError(f"Options field {field} is not valid for this backend")
        self._options.update_options(**fields)

    @property
    def options(self):
        """Return the options for the backend

        The options of a backend are the dynamic parameters defining
        how the backend is used. These are used to control the :meth:`run`
        method.
        """
        return self._options

    @property
    def provider(self):
        """Return the backend Provider.

        Returns:
            Provider: the Provider responsible for the backend.
        """
        return self._provider

    @abstractmethod
    def run(self, run_input, **options):
        """Run on the backend.

        This method returns a :class:`~qiskit.providers.Job` object
        that runs circuits. Depending on the backend this may be either an async
        or sync call. It is at the discretion of the provider to decide whether
        running should block until the execution is finished or not: the Job
        class can handle either situation.

        Args:
            run_input (QuantumCircuit or Schedule or ScheduleBlock or list): An
                individual or a list of :class:`.QuantumCircuit`,
                :class:`~qiskit.pulse.ScheduleBlock`, or :class:`~qiskit.pulse.Schedule` objects to
                run on the backend.
            options: Any kwarg options to pass to the backend for running the
                config. If a key is also present in the options
                attribute/object then the expectation is that the value
                specified will be used instead of what's set in the options
                object.

        Returns:
            Job: The job object for the run
        """
        pass
