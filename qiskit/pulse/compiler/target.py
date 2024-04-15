# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Compiler target information."""

from __future__ import annotations

from typing import Protocol
import dataclasses

from qiskit.pulse import model
from qiskit.pulse.exceptions import NotExistingComponent


class PulseTarget(Protocol):
    """Protocol of feeding the Qiskit pulse compiler with hardware configurations.

    This class defines interfaces for the Qiskit pulse compiler to receive
    necessary information to lower the input pulse programs.

    Because the system configuration is usually vendor dependent, we don't enforce
    the use of any Qiskit abstract class to give such vendors
    more flexibility to choose the most convenient data structure to
    represent their own devices.

    The collection of interfaces is designed to provide all necessary information
    to gurantee the builtin Qiskit pulse compiler passes can convert
    the Qiskit pulse program representation into a vendor specific target binary.
    Qiskit pulse uses an abstract model to represent hardware components, such as
    :class:`.QubitFrame` and :class:`qiskit.pulse.model.Qubit`.
    These components might be, not limited to, a software abstraction of
    a numerically controlled oscillator tracking the rotating frame of qubit,
    and a microwave or laser element connected with a particular qubit.

    This abstraction allows quantum physics experimentalists to adopt
    similar semantics to the familiar ciruict quantum electrodynamics theory
    without concerning about the detailed setup of the hardware device,
    while the Qiskit compiler needs to lower those components down to the actual equipment.

    Here we make native assumptions. A provider of this taregt must design
    its data structure to meet following rules otherwise the compiled program may cause
    faulty behavior. The hardware devices, or pulse controller,
    must distinghish every hardware ports and frames with unique string identifiers, and
    frames are tied to ports. In other words, frames don't exist standalone and
    always can form a mixed frame with a port in pair.

    In addition, Qiskit provides special preset frame types,
    :class:`.QubitFrame` and :class:`.MeasurementFrame`, and the backend must provide
    special string identifiers for these frame objects.
    If there is any architecture specific frames, a vendor can also define custom
    frame subclasses to extend the target model to align with their hardware devices.
    If the backend defines any calibration for its basis gates,
    we assume the same string identifier is used within the calibration so that
    a user provided inline calibration (pulse gates) can work seamlessly
    with the backend calibrated gate sets.

    See :class:`.QiskitPulseTarget` for the reference implementation.
    """

    def get_frame_identifier(
        self,
        frame: model.Frame,
    ) -> str:
        """Return the unique frame identifier of the Qiskit pulse frame.

        Args:
            frame: Qiskit pulse frame to inquire.

        Returns:
            A unique frame identifier.

        Raises:
            NotExistingComponent: When the frame identifier is not found.
        """
        raise NotExistingComponent(
            f"This hardware doesn't proivde any resource implementing {frame}."
        )

    def get_port_identifier(
        self,
        pulse_endpoint: model.PulseEndpoint,
        op_type: str,
    ) -> str:
        """Return the unique port identifier of the Qiskit Pulse pulse endpoint.

        Args:
            pulse_endpoint: Qiskit pulse PulseEndpoint object to inquire.
            op_type: Context of operation.

        Returns:
            A unique port identifier.

        Raises:
            NotExistingComponent: When the port identifier is not found.
        """
        raise NotExistingComponent(
            f"This hardware doesn't provide any resource implementing {pulse_endpoint}."
        )

    def reserved_mixed_frames(
        self,
        *,
        frame: model.Frame | None = None,
        pulse_endpoint: model.PulseEndpoint | None = None,
    ) -> list[model.MixedFrame]:
        """Return a list of mixed frames reserved for the backend gate calibrations.

        Args:
            frame: Qiskit pulse Frame object to include.
            pulse_endpoint: Qiskit pulse PulseEndpoint object to include.

        Returns:
            A list of Qiskit pulse mixed frame objects that
            include given frame and logical element.
        """
        return []


@dataclasses.dataclass(frozen=True)
class TXPort:
    """Software abstraction of transmission port.

    Transmission ports send control singals down to qubit
    or any quantum elements.
    """

    identifier: str
    """Unique identifier of this port."""

    qubits: tuple[int]
    """Tuple of qubit indicies that are affected through this port."""

    num_frames: int
    """Number of hadware supported frames tied to this port."""

    reserved_frames: list[str]
    """List of frame identifiers already used for circuit gate operations."""


@dataclasses.dataclass(frozen=True)
class RXPort:
    """Software abstraction of receiver port.

    Reciever ports record signals emitted from qubits
    to readout their state.
    """

    identifier: str
    """Unique identifier of this port."""

    qubits: tuple[int]
    """Tuple of qubit indices that this port get signals from."""


class ControlPort(TXPort):
    """A type of port controlling qubit state."""

    pass


class MeasurePort(TXPort):
    """A type of port stimulating qubits for state readout."""

    pass


class QiskitPulseTarget(PulseTarget):
    """Qiskit reference implementation of :class:`.PulseTarget`."""

    __type_alias = {
        ControlPort: "control",
        MeasurePort: "measure",
    }

    def __init__(
        self,
        qubit_frames: dict[int, str] | None = None,
        meas_frames: dict[int, str] | None = None,
        tx_ports: list[TXPort] | None = None,
        rx_ports: dict[str, dict] | None = None,
    ) -> None:
        """Create new Qiskit pulse target.

        Args:
            qubit_frames: A dictionary of qubit frame identifier keyed on
                the Qiskit :class:`.QubitFrame` object index.
                This frame must track the rotating frame
                of the qubit control signal.
            meas_frames: A dictionary of measurement frame identifier keyed on
                the Qiskit :class:`.MeasurementFrame` object index.
                This frame must track the rotating frame
                of the measurement stimulus signal.
            tx_ports: A list of :class:`.TXPort` dataclasses representing
                available hardware resources on the controller.
                TX port sends control singals down to qubit or any quantum elements.
                Port is not a user-configurable element.
            rx_ports: A list of dictionary representing a spec of receiver ports.

        TODO: implement RX port configuration, i.e. acquire port. Spec is not yet designed well.
        """
        self._qubit_frames: dict[int, str] = qubit_frames
        self._meas_frames: dict[int, str] = meas_frames
        self._calibrated_mixed_frames: list[model.MixedFrame] = []

        self._tx_ports: dict[str, dict[tuple[int, ...], TXPort]] = {
            t: {} for t in self.__type_alias.values()
        }

        # TODO what is the data?
        self._rx_ports = rx_ports

        # Cache all port names for efficient resource check.
        self._port_names = [p.identifier for p in tx_ports]

        self._sort_tx_ports(tx_ports=tx_ports)
        self._build_mixed_frames(tx_ports=tx_ports)

    def _sort_tx_ports(self, tx_ports: list[TXPort]):
        # Helper method to sort tx port list by port type and qubit indices.
        for port in tx_ports:
            try:
                port_type_str = self.__type_alias[type(port)]
            except KeyError as ex:
                raise TypeError(
                    f"Transmission port {port} is not defined type in {self.__class__.__name__}. "
                    "Please create new implementation of PulseTarget protocol for your contoller."
                ) from ex
            if (indices := tuple(port.qubits)) in self._tx_ports[port_type_str]:
                raise TypeError(
                    f"Transmission port '{port_type_str}' for qubits {indices} is already defined. "
                    "Qiskit reference implmentation of PulseTarget doesn't allow multiple ports "
                    "for the same set of qubits under under same port type."
                )
            self._tx_ports[port_type_str][indices] = port

    def _build_mixed_frames(self, tx_ports: list[TXPort]):
        # Helper method to build Qiskit MixedFrame models from the port configuraion data.
        qubit_frames_inv = dict(zip(self._qubit_frames.values(), self._qubit_frames.keys()))
        meas_frames_inv = dict(zip(self._meas_frames.values(), self._meas_frames.keys()))
        for port in tx_ports:
            for fuid in port.reserved_frames:
                if (fidx := qubit_frames_inv.get(fuid, None)) is not None:
                    # Qiskit special type: Qubit mixed frame
                    mixed_frame = model.MixedFrame(
                        pulse_target=model.Qubit(port.qubits[0]),
                        frame=model.QubitFrame(fidx),
                    )
                elif (fidx := meas_frames_inv.get(fuid, None)) is not None:
                    # Qiskit special type: Measurement mixed frame
                    mixed_frame = model.MixedFrame(
                        pulse_target=model.Qubit(port.qubits[0]),
                        frame=model.MeasurementFrame(fidx),
                    )
                else:
                    # Generic
                    mixed_frame = model.MixedFrame(
                        pulse_target=model.Port(port.identifier),
                        frame=model.GenericFrame(fuid),
                    )
                self._calibrated_mixed_frames.append(mixed_frame)

    def get_frame_identifier(
        self,
        frame: model.Frame,
    ) -> str:
        if isinstance(frame, model.GenericFrame):
            return frame.name
        if isinstance(frame, model.QubitFrame):
            try:
                return self._qubit_frames[frame.index]
            except KeyError as ex:
                raise NotExistingComponent(
                    "This control system doesn't provide any frame for "
                    f"implementing QubitFrame of index {frame.index}."
                ) from ex
        if isinstance(frame, model.MeasurementFrame):
            try:
                return self._meas_frames[frame.index]
            except KeyError as ex:
                raise NotExistingComponent(
                    "This control system doesn't provide any frame for "
                    f"implementing MeasurementFrame of index {frame.index}."
                ) from ex
        raise TypeError(
            f"{self.__class__.__name__} doesn't recognize the frame object of type "
            f"{frame.__class__.__name__}. If you are using a custom subclass, "
            "you must also define PulseTarget class supporting this type."
        )

    def get_port_identifier(
        self,
        pulse_endpoint: model.PulseTarget,
        op_type: str,
    ) -> str:
        if isinstance(pulse_endpoint, model.Port):
            if pulse_endpoint.name not in self._port_names:
                raise NotExistingComponent(
                    f"Port identifier {pulse_endpoint.name} is not defined in this system. "
                    "Hardware may not implement this port."
                )
            return pulse_endpoint.name

        # Resolve logical elements.
        if op_type not in self._tx_ports:
            raise ValueError(
                f"Given operation type '{op_type}' is not defined on this system. "
                f"PulseTarget of the controller supports {list(self.__type_alias.keys())} types. "
                f"Request of port identifier for {pulse_endpoint} "
                f"in the '{op_type}' context is not resolved."
            )
        qubit_index = tuple(pulse_endpoint.index)
        try:
            return self._tx_ports[op_type][qubit_index].identifier
        except KeyError as ex:
            raise NotExistingComponent(
                "This control system doesn't provide any hardware port for "
                f"implementing LogicalElement {pulse_endpoint} in the '{op_type}' context."
            ) from ex

    def reserved_mixed_frames(
        self,
        *,
        frame: model.Frame | None = None,
        pulse_endpoint: model.PulseEndpoint | None = None,
    ) -> list[model.MixedFrame]:
        if frame is None and pulse_endpoint is None:
            raise ValueError("Either frame or pulse_endpoint must be specified.")

        out = []
        for mixed_frame in self._calibrated_mixed_frames:
            if frame is not None and frame != mixed_frame.frame:
                continue
            if pulse_endpoint is not None and pulse_endpoint != mixed_frame.pulse_target:
                continue
            out.append(mixed_frame)
        return out
