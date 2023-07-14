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

"""Matplotlib plotter API."""

from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import core, drawings, types
from qiskit.visualization.pulse_v2.plotters.base_plotter import BasePlotter
from qiskit.visualization.utils import matplotlib_close_if_inline


class Mpl2DPlotter(BasePlotter):
    """Matplotlib API for pulse drawer.

    This plotter places canvas charts along y axis of 2D canvas with vertical offset.
    Each chart is map to X-Y axis of the canvas.
    """

    def __init__(self, canvas: core.DrawerCanvas, axis: Optional[plt.Axes] = None):
        """Create new plotter.

        Args:
            canvas: Configured drawer canvas object. Canvas object should be updated
                with `.update` method before set to the plotter API.
            axis: Matplotlib axis object. When `axis` is provided, the plotter updates
                given axis instead of creating and returning new matplotlib figure.
        """
        super().__init__(canvas=canvas)

        # calculate height of all charts
        canvas_height = 0
        for chart in self.canvas.charts:
            if not chart.is_active and not self.canvas.formatter["control.show_empty_channel"]:
                continue
            canvas_height += chart.vmax - chart.vmin
        # set min canvas_height size
        canvas_height = max(canvas_height, 0.1)

        if axis is None:
            fig_h = canvas_height * self.canvas.formatter["general.fig_chart_height"]
            fig_w = self.canvas.formatter["general.fig_width"]

            self.figure = plt.figure(figsize=(fig_w, fig_h))
            self.ax = self.figure.add_subplot(1, 1, 1)
        else:
            self.figure = axis.figure
            self.ax = axis

        self.initialize_canvas()

    def initialize_canvas(self):
        """Format appearance of matplotlib canvas."""
        self.ax.set_facecolor(self.canvas.formatter["color.background"])

        # axis labels
        self.ax.set_yticklabels([])
        self.ax.yaxis.set_tick_params(left=False)

    def draw(self):
        """Output drawings stored in canvas object."""
        # axis configuration
        axis_config = self.canvas.layout["time_axis_map"](
            time_window=self.canvas.time_range,
            axis_breaks=self.canvas.time_breaks,
            dt=self.canvas.device.dt,
        )

        current_y = 0
        margin_y = self.canvas.formatter["margin.between_channel"]
        for chart in self.canvas.charts:
            if not chart.is_active and not self.canvas.formatter["control.show_empty_channel"]:
                continue
            current_y -= chart.vmax
            for _, data in chart.collections:
                # calculate scaling factor
                if not data.ignore_scaling:
                    # product of channel-wise scaling and chart level scaling
                    scale = max(self.canvas.chan_scales.get(chan, 1.0) for chan in data.channels)
                    scale *= chart.scale
                else:
                    scale = 1.0

                x = data.xvals
                y = scale * data.yvals + current_y

                if isinstance(data, drawings.LineData):
                    # line object
                    if data.fill:
                        self.ax.fill_between(x, y1=y, y2=current_y * np.ones_like(y), **data.styles)
                    else:
                        self.ax.plot(x, y, **data.styles)
                elif isinstance(data, drawings.TextData):
                    # text object
                    text = rf"${data.latex}$" if data.latex else data.text
                    # replace dynamic text
                    text = text.replace(types.DynamicString.SCALE, f"{chart.scale:.1f}")
                    self.ax.text(x=x[0], y=y[0], s=text, **data.styles)
                elif isinstance(data, drawings.BoxData):
                    xy = x[0], y[0]
                    box = Rectangle(
                        xy, width=x[1] - x[0], height=y[1] - y[0], fill=True, **data.styles
                    )
                    self.ax.add_patch(box)
                else:
                    raise VisualizationError(
                        "Data {name} is not supported "
                        "by {plotter}".format(name=data, plotter=self.__class__.__name__)
                    )
            # axis break
            for pos in axis_config.axis_break_pos:
                self.ax.text(
                    x=pos,
                    y=current_y,
                    s="//",
                    ha="center",
                    va="center",
                    zorder=self.canvas.formatter["layer.axis_label"],
                    fontsize=self.canvas.formatter["text_size.axis_break_symbol"],
                    rotation=180,
                )

            # shift chart position
            current_y += chart.vmin - margin_y

        # remove the last margin
        current_y += margin_y

        y_max = self.canvas.formatter["margin.top"]
        y_min = current_y - self.canvas.formatter["margin.bottom"]

        # plot axis break line
        for pos in axis_config.axis_break_pos:
            self.ax.plot(
                [pos, pos],
                [y_min, y_max],
                zorder=self.canvas.formatter["layer.fill_waveform"] + 1,
                linewidth=self.canvas.formatter["line_width.axis_break"],
                color=self.canvas.formatter["color.background"],
            )

        # label
        self.ax.set_xticks(list(axis_config.axis_map.keys()))
        self.ax.set_xticklabels(
            list(axis_config.axis_map.values()),
            fontsize=self.canvas.formatter["text_size.axis_label"],
        )
        self.ax.set_xlabel(
            axis_config.label, fontsize=self.canvas.formatter["text_size.axis_label"]
        )

        # boundary
        if axis_config.window == (0, 0):
            self.ax.set_xlim(0, 1)
        else:
            self.ax.set_xlim(*axis_config.window)
        self.ax.set_ylim(y_min, y_max)

        # title
        if self.canvas.fig_title:
            self.ax.text(
                x=axis_config.window[0],
                y=y_max,
                s=self.canvas.fig_title,
                ha="left",
                va="bottom",
                zorder=self.canvas.formatter["layer.fig_title"],
                color=self.canvas.formatter["color.fig_title"],
                size=self.canvas.formatter["text_size.fig_title"],
            )

    def get_image(self, interactive: bool = False) -> mpl.pyplot.Figure:
        """Get image data to return.

        Args:
            interactive: When set `True` show the circuit in a new window.
                This depends on the matplotlib backend being used supporting this.

        Returns:
            Matplotlib figure data.
        """
        matplotlib_close_if_inline(self.figure)

        if self.figure and interactive:
            self.figure.show()

        return self.figure
