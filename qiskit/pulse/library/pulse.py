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

"""Pulses are descriptions of waveform envelopes. They can be transmitted by control electronics
to the device.
"""
from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from typing import Any
from qiskit.utils.deprecate_pulse import deprecate_pulse_func

from qiskit.circuit.parameterexpression import ParameterExpression


if typing.TYPE_CHECKING:
    from qiskit.providers import Backend  # pylint: disable=cyclic-import


class Pulse(ABC):
    """The abstract superclass for pulses. Pulses are complex-valued waveform envelopes. The
    modulation phase and frequency are specified separately from ``Pulse``s.
    """

    __slots__ = ("duration", "name", "_limit_amplitude")

    limit_amplitude = True

    @abstractmethod
    @deprecate_pulse_func
    def __init__(
        self,
        duration: int | ParameterExpression,
        name: str | None = None,
        limit_amplitude: bool | None = None,
    ):
        """Abstract base class for pulses
        Args:
            duration: Duration of the pulse
            name: Optional name for the pulse
            limit_amplitude: If ``True``, then limit the amplitude of the waveform to 1.
                             The default value of ``None`` causes the flag value to be
                             derived from :py:attr:`~limit_amplitude` which is ``True``
                             by default but may be set by the user to disable amplitude
                             checks globally.
        """
        if limit_amplitude is None:
            limit_amplitude = self.__class__.limit_amplitude

        self.duration = duration
        self.name = name
        self._limit_amplitude = limit_amplitude

    @property
    def id(self) -> int:  # pylint: disable=invalid-name
        """Unique identifier for this pulse."""
        return id(self)

    @property
    @abstractmethod
    def parameters(self) -> dict[str, typing.Any]:
        """Return a dictionary containing the pulse's parameters."""
        pass

    def is_parameterized(self) -> bool:
        """Return True iff the instruction is parameterized."""
        raise NotImplementedError

    def draw(
        self,
        style: dict[str, Any] | None = None,
        backend: Backend | None = None,
        time_range: tuple[int, int] | None = None,
        time_unit: str = "dt",
        show_waveform_info: bool = True,
        plotter: str = "mpl2d",
        axis: Any | None = None,
    ):
        """Plot the interpolated envelope of pulse.

        Args:
            style: Stylesheet options. This can be dictionary or preset stylesheet classes. See
                :py:class:`~qiskit.visualization.pulse_v2.stylesheets.IQXStandard`,
                :py:class:`~qiskit.visualization.pulse_v2.stylesheets.IQXSimple`, and
                :py:class:`~qiskit.visualization.pulse_v2.stylesheets.IQXDebugging` for details of
                preset stylesheets.
            backend (Optional[BaseBackend]): Backend object to play the input pulse program.
                If provided, the plotter may use to make the visualization hardware aware.
            time_range: Set horizontal axis limit. Tuple ``(tmin, tmax)``.
            time_unit: The unit of specified time range either ``dt`` or ``ns``.
                The unit of ``ns`` is available only when ``backend`` object is provided.
            show_waveform_info: Show waveform annotations, i.e. name, of waveforms.
                Set ``True`` to show additional information about waveforms.
            plotter: Name of plotter API to generate an output image.
                One of following APIs should be specified::

                    mpl2d: Matplotlib API for 2D image generation.
                        Matplotlib API to generate 2D image. Charts are placed along y axis with
                        vertical offset. This API takes matplotlib.axes.Axes as `axis` input.

                `axis` and `style` kwargs may depend on the plotter.
            axis: Arbitrary object passed to the plotter. If this object is provided,
                the plotters use a given ``axis`` instead of internally initializing
                a figure object. This object format depends on the plotter.
                See plotter argument for details.

        Returns:
            Visualization output data.
            The returned data type depends on the ``plotter``.
            If matplotlib family is specified, this will be a ``matplotlib.pyplot.Figure`` data.
        """
        # pylint: disable=cyclic-import
        from qiskit.visualization import pulse_drawer

        return pulse_drawer(
            program=self,
            style=style,
            backend=backend,
            time_range=time_range,
            time_unit=time_unit,
            show_waveform_info=show_waveform_info,
            plotter=plotter,
            axis=axis,
        )

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pulse):
            return NotImplemented
        return isinstance(other, type(self))

    @abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError
