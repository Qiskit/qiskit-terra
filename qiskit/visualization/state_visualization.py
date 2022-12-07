# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name
# pylint: disable=missing-param-doc,missing-type-doc,unused-argument

"""
Visualization functions for quantum states.
"""

from typing import Optional, List, Union
from functools import reduce
import colorsys
import numpy as np
from qiskit import user_config
from qiskit.quantum_info.states.statevector import Statevector
from qiskit.quantum_info.operators.symplectic import PauliList, SparsePauliOp
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.utils.deprecation import deprecate_arguments
from qiskit.utils import optionals as _optionals
from qiskit.circuit.tools.pi_check import pi_check

from .array import array_to_latex
from .utils import matplotlib_close_if_inline
from .exceptions import VisualizationError


@deprecate_arguments({"rho": "state"})
@_optionals.HAS_MATPLOTLIB.require_in_call
def plot_state_hinton(
    state, title="", figsize=None, ax_real=None, ax_imag=None, *, rho=None, filename=None
):
    """Plot a hinton diagram for the density matrix of a quantum state.

    The hinton diagram represents the values of a matrix using
    squares, whose size indicate the magnitude of their corresponding value
    and their color, its sign. A white square means the value is positive and
    a black one means negative.

    Args:
        state (Statevector or DensityMatrix or ndarray): An N-qubit quantum state.
        title (str): a string that represents the plot title
        figsize (tuple): Figure size in inches.
        filename (str): file path to save image to.
        ax_real (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified a new matplotlib
            Figure will be created and used. If this is specified without an
            ax_imag only the real component plot will be generated.
            Additionally, if specified there will be no returned Figure since
            it is redundant.
        ax_imag (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified a new matplotlib
            Figure will be created and used. If this is specified without an
            ax_imag only the real component plot will be generated.
            Additionally, if specified there will be no returned Figure since
            it is redundant.

    Returns:
         matplotlib.Figure:
            The matplotlib.Figure of the visualization if
            neither ax_real or ax_imag is set.

    Raises:
        MissingOptionalLibraryError: Requires matplotlib.
        VisualizationError: if input is not a valid N-qubit state.

    Examples:
        .. jupyter-execute::

            import numpy as np
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import DensityMatrix
            from qiskit.visualization import plot_state_hinton

            qc = QuantumCircuit(2)
            qc.h([0, 1])
            qc.cz(0,1)
            qc.ry(np.pi/3 , 0)
            qc.rx(np.pi/5, 1)

            state = DensityMatrix(qc)
            plot_state_hinton(state, title="New Hinton Plot")

    """
    from matplotlib import pyplot as plt

    # Figure data
    rho = DensityMatrix(state)
    num = rho.num_qubits
    if num is None:
        raise VisualizationError("Input is not a multi-qubit quantum state.")
    max_weight = 2 ** np.ceil(np.log(np.abs(rho.data).max()) / np.log(2))
    datareal = np.real(rho.data)
    dataimag = np.imag(rho.data)

    if figsize is None:
        figsize = (8, 5)
    if not ax_real and not ax_imag:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        if ax_real:
            fig = ax_real.get_figure()
        else:
            fig = ax_imag.get_figure()
        ax1 = ax_real
        ax2 = ax_imag
    # Reversal is to account for Qiskit's endianness.
    column_names = [bin(i)[2:].zfill(num) for i in range(2**num)]
    row_names = [bin(i)[2:].zfill(num) for i in range(2**num)][::-1]
    ly, lx = datareal.shape
    # Real
    if ax1:
        ax1.patch.set_facecolor("gray")
        ax1.set_aspect("equal", "box")
        ax1.xaxis.set_major_locator(plt.NullLocator())
        ax1.yaxis.set_major_locator(plt.NullLocator())

        for (x, y), w in np.ndenumerate(datareal):
            # Convert from matrix co-ordinates to plot co-ordinates.
            plot_x, plot_y = y, lx - x - 1
            color = "white" if w > 0 else "black"
            size = np.sqrt(np.abs(w) / max_weight)
            rect = plt.Rectangle(
                [0.5 + plot_x - size / 2, 0.5 + plot_y - size / 2],
                size,
                size,
                facecolor=color,
                edgecolor=color,
            )
            ax1.add_patch(rect)

        ax1.set_xticks(0.5 + np.arange(lx))
        ax1.set_yticks(0.5 + np.arange(ly))
        ax1.set_xlim([0, lx])
        ax1.set_ylim([0, ly])
        ax1.set_yticklabels(row_names, fontsize=14)
        ax1.set_xticklabels(column_names, fontsize=14, rotation=90)
        ax1.set_title("Re[$\\rho$]", fontsize=14)
    # Imaginary
    if ax2:
        ax2.patch.set_facecolor("gray")
        ax2.set_aspect("equal", "box")
        ax2.xaxis.set_major_locator(plt.NullLocator())
        ax2.yaxis.set_major_locator(plt.NullLocator())

        for (x, y), w in np.ndenumerate(dataimag):
            # Convert from matrix co-ordinates to plot co-ordinates.
            plot_x, plot_y = y, lx - x - 1
            color = "white" if w > 0 else "black"
            size = np.sqrt(np.abs(w) / max_weight)
            rect = plt.Rectangle(
                [0.5 + plot_x - size / 2, 0.5 + plot_y - size / 2],
                size,
                size,
                facecolor=color,
                edgecolor=color,
            )
            ax2.add_patch(rect)

        ax2.set_xticks(0.5 + np.arange(lx))
        ax2.set_yticks(0.5 + np.arange(ly))
        ax2.set_xlim([0, lx])
        ax2.set_ylim([0, ly])
        ax2.set_yticklabels(row_names, fontsize=14)
        ax2.set_xticklabels(column_names, fontsize=14, rotation=90)
        ax2.set_title("Im[$\\rho$]", fontsize=14)
    fig.tight_layout()
    if title:
        fig.suptitle(title, fontsize=16)
    if ax_real is None and ax_imag is None:
        matplotlib_close_if_inline(fig)
    if filename is None:
        return fig
    else:
        return fig.savefig(filename)


@_optionals.HAS_MATPLOTLIB.require_in_call
def plot_bloch_vector(
    bloch,
    title="",
    ax=None,
    figsize=None,
    coord_type="cartesian",
    **kwargs,
):
    """Plot the Bloch sphere.

    Plot a Bloch sphere with the specified coordinates, that can be given in both
    cartesian and spherical systems.

    Args:
        bloch (list[double]): array of three elements where [<x>, <y>, <z>] (Cartesian)
            or [<r>, <theta>, <phi>] (spherical in radians)
            <theta> is inclination angle from +z direction
            <phi> is azimuth from +x direction
        title (str): a string that represents the plot title
        ax (matplotlib.axes.Axes): An Axes to use for rendering the bloch
            sphere
        figsize (tuple): Figure size in inches. Has no effect is passing ``ax``.
        coord_type (str): a string that specifies coordinate type for bloch
            (Cartesian or spherical), default is Cartesian
        **kwargs:
                Options as for matplotlib.patches.FancyArrowPatch

    Returns:
        Figure: A matplotlib figure instance if ``ax = None``.

    Raises:
        MissingOptionalLibraryError: Requires matplotlib.

    Examples:
        .. jupyter-execute::

           from qiskit.visualization import plot_bloch_vector

           plot_bloch_vector([0,1,0], title="New Bloch Sphere")

        .. jupyter-execute::

           # You can use spherical coordinates instead of cartesian.

           import numpy as np

           plot_bloch_vector([1, np.pi/2, np.pi/3], coord_type='spherical')

    """
    from .bloch import Bloch

    if figsize is None:
        figsize = (5, 5)
    B = Bloch(axes=ax)
    if coord_type == "spherical":
        r, theta, phi = bloch[0], bloch[1], bloch[2]
        bloch[0] = r * np.sin(theta) * np.cos(phi)
        bloch[1] = r * np.sin(theta) * np.sin(phi)
        bloch[2] = r * np.cos(theta)
    B.add_vectors(bloch, **kwargs)
    B.render(title=title)
    if ax is None:
        fig = B.fig
        fig.set_size_inches(figsize[0], figsize[1])
        matplotlib_close_if_inline(fig)
        return fig
    return None


@deprecate_arguments({"rho": "state"})
@_optionals.HAS_MATPLOTLIB.require_in_call
def plot_bloch_multivector(
    state,
    title="",
    figsize=None,
    *,
    rho=None,
    reverse_bits=False,
    filename=None,
    **kwargs,
):
    r"""Plot a Bloch sphere for each qubit.

    Each component :math:`(x,y,z)` of the Bloch sphere labeled as 'qubit i' represents the expected
    value of the corresponding Pauli operator acting only on that qubit, that is, the expected value
    of :math:`I_{N-1} \otimes\dotsb\otimes I_{i+1}\otimes P_i \otimes I_{i-1}\otimes\dotsb\otimes
    I_0`, where :math:`N` is the number of qubits, :math:`P\in \{X,Y,Z\}` and :math:`I` is the
    identity operator.

    Args:
        state (Statevector or DensityMatrix or ndarray): an N-qubit quantum state.
        title (str): a string that represents the plot title
        figsize (tuple): Has no effect, here for compatibility only.
        reverse_bits (bool): If True, plots qubits following Qiskit's convention [Default:False].
        **kwargs:
                Options as for matplotlib.patches.FancyArrowPatch

    Returns:
        matplotlib.Figure:
            A matplotlib figure instance.

    Raises:
        MissingOptionalLibraryError: Requires matplotlib.
        VisualizationError: if input is not a valid N-qubit state.

    Examples:
        .. jupyter-execute::

            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
            from qiskit.visualization import plot_bloch_multivector

            qc = QuantumCircuit(2)
            qc.h(0)
            qc.x(1)

            state = Statevector(qc)
            plot_bloch_multivector(state)

        .. jupyter-execute::

           # You can reverse the order of the qubits.

           from qiskit.quantum_info import DensityMatrix

           qc = QuantumCircuit(2)
           qc.h([0, 1])
           qc.t(1)
           qc.s(0)
           qc.cx(0,1)

           matrix = DensityMatrix(qc)
           plot_bloch_multivector(matrix, title='My Bloch Spheres', reverse_bits=True)

    """
    from matplotlib import pyplot as plt

    # Data
    bloch_data = (
        _bloch_multivector_data(state)[::-1] if reverse_bits else _bloch_multivector_data(state)
    )
    num = len(bloch_data)
    width, height = plt.figaspect(1 / num)
    fig = plt.figure(figsize=(width, height))
    for i in range(num):
        pos = num - 1 - i if reverse_bits else i
        ax = fig.add_subplot(1, num, i + 1, projection="3d")
        plot_bloch_vector(
            bloch_data[i],
            "qubit " + str(pos),
            ax=ax,
            figsize=figsize,
            **kwargs,
        )
    fig.suptitle(title, fontsize=16, y=1.01)
    matplotlib_close_if_inline(fig)
    if filename is None:
        return fig
    else:
        return fig.savefig(filename)


@deprecate_arguments({"rho": "state"})
@_optionals.HAS_MATPLOTLIB.require_in_call
def plot_state_city(
    state,
    title="",
    figsize=None,
    color=None,
    alpha=1,
    ax_real=None,
    ax_imag=None,
    *,
    rho=None,
    filename=None,
):
    """Plot the cityscape of quantum state.

    Plot two 3d bar graphs (two dimensional) of the real and imaginary
    part of the density matrix rho.

    Args:
        state (Statevector or DensityMatrix or ndarray): an N-qubit quantum state.
        title (str): a string that represents the plot title
        figsize (tuple): Figure size in inches.
        color (list): A list of len=2 giving colors for real and
            imaginary components of matrix elements.
        alpha (float): Transparency value for bars
        ax_real (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified a new matplotlib
            Figure will be created and used. If this is specified without an
            ax_imag only the real component plot will be generated.
            Additionally, if specified there will be no returned Figure since
            it is redundant.
        ax_imag (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified a new matplotlib
            Figure will be created and used. If this is specified without an
            ax_real only the imaginary component plot will be generated.
            Additionally, if specified there will be no returned Figure since
            it is redundant.

    Returns:
         matplotlib.Figure:
            The matplotlib.Figure of the visualization if the
            ``ax_real`` and ``ax_imag`` kwargs are not set

    Raises:
        MissingOptionalLibraryError: Requires matplotlib.
        ValueError: When 'color' is not a list of len=2.
        VisualizationError: if input is not a valid N-qubit state.

    Examples:
        .. jupyter-execute::

           # You can choose different colors for the real and imaginary parts of the density matrix.

           from qiskit import QuantumCircuit
           from qiskit.quantum_info import DensityMatrix
           from qiskit.visualization import plot_state_city

           qc = QuantumCircuit(2)
           qc.h(0)
           qc.cx(0, 1)

           state = DensityMatrix(qc)
           plot_state_city(state, color=['midnightblue', 'crimson'], title="New State City")

        .. jupyter-execute::

           # You can make the bars more transparent to better see the ones that are behind
           # if they overlap.

           import numpy as np
           from qiskit.quantum_info import Statevector

           qc = QuantumCircuit(2)
           qc.h([0, 1])
           qc.cz(0,1)
           qc.ry(np.pi/3, 0)
           qc.rx(np.pi/5, 1)

           state = Statevector(qc)
           plot_state_city(state, alpha=0.6)

    """
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    rho = DensityMatrix(state)
    num = rho.num_qubits
    if num is None:
        raise VisualizationError("Input is not a multi-qubit quantum state.")

    # get the real and imag parts of rho
    datareal = np.real(rho.data)
    dataimag = np.imag(rho.data)

    # get the labels
    column_names = [bin(i)[2:].zfill(num) for i in range(2**num)]
    row_names = [bin(i)[2:].zfill(num) for i in range(2**num)]

    lx = len(datareal[0])  # Work out matrix dimensions
    ly = len(datareal[:, 0])
    xpos = np.arange(0, lx, 1)  # Set up a mesh of positions
    ypos = np.arange(0, ly, 1)
    xpos, ypos = np.meshgrid(xpos + 0.25, ypos + 0.25)

    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros(lx * ly)

    dx = 0.5 * np.ones_like(zpos)  # width of bars
    dy = dx.copy()
    dzr = datareal.flatten()
    dzi = dataimag.flatten()

    if color is None:
        color = ["#648fff", "#648fff"]
    else:
        if len(color) != 2:
            raise ValueError("'color' must be a list of len=2.")
        if color[0] is None:
            color[0] = "#648fff"
        if color[1] is None:
            color[1] = "#648fff"
    if ax_real is None and ax_imag is None:
        # set default figure size
        if figsize is None:
            figsize = (15, 5)

        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    elif ax_real is not None:
        fig = ax_real.get_figure()
        ax1 = ax_real
        ax2 = ax_imag
    else:
        fig = ax_imag.get_figure()
        ax1 = None
        ax2 = ax_imag

    max_dzr = max(dzr)
    min_dzr = min(dzr)
    min_dzi = np.min(dzi)
    max_dzi = np.max(dzi)

    if ax1 is not None:
        fc1 = generate_facecolors(xpos, ypos, zpos, dx, dy, dzr, color[0])
        for idx, cur_zpos in enumerate(zpos):
            if dzr[idx] > 0:
                zorder = 2
            else:
                zorder = 0
            b1 = ax1.bar3d(
                xpos[idx],
                ypos[idx],
                cur_zpos,
                dx[idx],
                dy[idx],
                dzr[idx],
                alpha=alpha,
                zorder=zorder,
            )
            b1.set_facecolors(fc1[6 * idx : 6 * idx + 6])

        xlim, ylim = ax1.get_xlim(), ax1.get_ylim()
        x = [xlim[0], xlim[1], xlim[1], xlim[0]]
        y = [ylim[0], ylim[0], ylim[1], ylim[1]]
        z = [0, 0, 0, 0]
        verts = [list(zip(x, y, z))]

        pc1 = Poly3DCollection(verts, alpha=0.15, facecolor="k", linewidths=1, zorder=1)

        if min(dzr) < 0 < max(dzr):
            ax1.add_collection3d(pc1)
        ax1.set_xticks(np.arange(0.5, lx + 0.5, 1))
        ax1.set_yticks(np.arange(0.5, ly + 0.5, 1))
        if max_dzr != min_dzr:
            ax1.axes.set_zlim3d(np.min(dzr), max(np.max(dzr) + 1e-9, max_dzi))
        else:
            if min_dzr == 0:
                ax1.axes.set_zlim3d(np.min(dzr), max(np.max(dzr) + 1e-9, np.max(dzi)))
            else:
                ax1.axes.set_zlim3d(auto=True)
        ax1.get_autoscalez_on()
        ax1.xaxis.set_ticklabels(row_names, fontsize=14, rotation=45, ha="right", va="top")
        ax1.yaxis.set_ticklabels(column_names, fontsize=14, rotation=-22.5, ha="left", va="center")
        ax1.set_zlabel("Re[$\\rho$]", fontsize=14)
        for tick in ax1.zaxis.get_major_ticks():
            tick.label1.set_fontsize(14)

    if ax2 is not None:
        fc2 = generate_facecolors(xpos, ypos, zpos, dx, dy, dzi, color[1])
        for idx, cur_zpos in enumerate(zpos):
            if dzi[idx] > 0:
                zorder = 2
            else:
                zorder = 0
            b2 = ax2.bar3d(
                xpos[idx],
                ypos[idx],
                cur_zpos,
                dx[idx],
                dy[idx],
                dzi[idx],
                alpha=alpha,
                zorder=zorder,
            )
            b2.set_facecolors(fc2[6 * idx : 6 * idx + 6])

        xlim, ylim = ax2.get_xlim(), ax2.get_ylim()
        x = [xlim[0], xlim[1], xlim[1], xlim[0]]
        y = [ylim[0], ylim[0], ylim[1], ylim[1]]
        z = [0, 0, 0, 0]
        verts = [list(zip(x, y, z))]

        pc2 = Poly3DCollection(verts, alpha=0.2, facecolor="k", linewidths=1, zorder=1)

        if min(dzi) < 0 < max(dzi):
            ax2.add_collection3d(pc2)
        ax2.set_xticks(np.arange(0.5, lx + 0.5, 1))
        ax2.set_yticks(np.arange(0.5, ly + 0.5, 1))
        if min_dzi != max_dzi:
            eps = 0
            ax2.axes.set_zlim3d(np.min(dzi), max(np.max(dzr) + 1e-9, np.max(dzi) + eps))
        else:
            if min_dzi == 0:
                ax2.set_zticks([0])
                eps = 1e-9
                ax2.axes.set_zlim3d(np.min(dzi), max(np.max(dzr) + 1e-9, np.max(dzi) + eps))
            else:
                ax2.axes.set_zlim3d(auto=True)

        ax2.xaxis.set_ticklabels(row_names, fontsize=14, rotation=45, ha="right", va="top")
        ax2.yaxis.set_ticklabels(column_names, fontsize=14, rotation=-22.5, ha="left", va="center")
        ax2.set_zlabel("Im[$\\rho$]", fontsize=14)
        for tick in ax2.zaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
        ax2.get_autoscalez_on()

    fig.suptitle(title, fontsize=16)
    if ax_real is None and ax_imag is None:
        matplotlib_close_if_inline(fig)
    if filename is None:
        return fig
    else:
        return fig.savefig(filename)


@deprecate_arguments({"rho": "state"})
@_optionals.HAS_MATPLOTLIB.require_in_call
def plot_state_paulivec(
    state, title="", figsize=None, color=None, ax=None, *, rho=None, filename=None
):
    """Plot the paulivec representation of a quantum state.

    Plot a bargraph of the mixed state rho over the pauli matrices

    Args:
        state (Statevector or DensityMatrix or ndarray): an N-qubit quantum state.
        title (str): a string that represents the plot title
        figsize (tuple): Figure size in inches.
        color (list or str): Color of the expectation value bars.
        ax (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified a new matplotlib
            Figure will be created and used. Additionally, if specified there
            will be no returned Figure since it is redundant.

    Returns:
         matplotlib.Figure:
            The matplotlib.Figure of the visualization if the
            ``ax`` kwarg is not set

    Raises:
        MissingOptionalLibraryError: Requires matplotlib.
        VisualizationError: if input is not a valid N-qubit state.

    Examples:
        .. jupyter-execute::

           # You can set a color for all the bars.

           from qiskit import QuantumCircuit
           from qiskit.quantum_info import Statevector
           from qiskit.visualization import plot_state_paulivec

           qc = QuantumCircuit(2)
           qc.h(0)
           qc.cx(0, 1)

           state = Statevector(qc)
           plot_state_paulivec(state, color='midnightblue', title="New PauliVec plot")

        .. jupyter-execute::

           # If you introduce a list with less colors than bars, the color of the bars will
           # alternate following the sequence from the list.

           import numpy as np
           from qiskit.quantum_info import DensityMatrix

           qc = QuantumCircuit(2)
           qc.h([0, 1])
           qc.cz(0, 1)
           qc.ry(np.pi/3, 0)
           qc.rx(np.pi/5, 1)

           matrix = DensityMatrix(qc)
           plot_state_paulivec(matrix, color=['crimson', 'midnightblue', 'seagreen'])
    """
    from matplotlib import pyplot as plt

    labels, values = _paulivec_data(state)
    numelem = len(values)

    if figsize is None:
        figsize = (7, 5)
    if color is None:
        color = "#648fff"

    ind = np.arange(numelem)  # the x locations for the groups
    width = 0.5  # the width of the bars
    if ax is None:
        return_fig = True
        fig, ax = plt.subplots(figsize=figsize)
    else:
        return_fig = False
        fig = ax.get_figure()
    ax.grid(zorder=0, linewidth=1, linestyle="--")
    ax.bar(ind, values, width, color=color, zorder=2)
    ax.axhline(linewidth=1, color="k")
    # add some text for labels, title, and axes ticks
    ax.set_ylabel("Expectation value", fontsize=14)
    ax.set_xticks(ind)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(labels, fontsize=14, rotation=70)
    ax.set_xlabel("Pauli", fontsize=14)
    ax.set_ylim([-1, 1])
    ax.set_facecolor("#eeeeee")
    for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
    ax.set_title(title, fontsize=16)
    if return_fig:
        matplotlib_close_if_inline(fig)
    if filename is None:
        return fig
    else:
        return fig.savefig(filename)


def n_choose_k(n, k):
    """Return the number of combinations for n choose k.

    Args:
        n (int): the total number of options .
        k (int): The number of elements.

    Returns:
        int: returns the binomial coefficient
    """
    if n == 0:
        return 0
    return reduce(lambda x, y: x * y[0] / y[1], zip(range(n - k + 1, n + 1), range(1, k + 1)), 1)


def lex_index(n, k, lst):
    """Return  the lex index of a combination..

    Args:
        n (int): the total number of options .
        k (int): The number of elements.
        lst (list): list

    Returns:
        int: returns int index for lex order

    Raises:
        VisualizationError: if length of list is not equal to k
    """
    if len(lst) != k:
        raise VisualizationError("list should have length k")
    comb = list(map(lambda x: n - 1 - x, lst))
    dualm = sum(n_choose_k(comb[k - 1 - i], i + 1) for i in range(k))
    return int(dualm)


def bit_string_index(s):
    """Return the index of a string of 0s and 1s."""
    n = len(s)
    k = s.count("1")
    if s.count("0") != n - k:
        raise VisualizationError("s must be a string of 0 and 1")
    ones = [pos for pos, char in enumerate(s) if char == "1"]
    return lex_index(n, k, ones)


def phase_to_rgb(complex_number):
    """Map a phase of a complexnumber to a color in (r,g,b).

    complex_number is phase is first mapped to angle in the range
    [0, 2pi] and then to the HSL color wheel
    """
    angles = (np.angle(complex_number) + (np.pi * 5 / 4)) % (np.pi * 2)
    rgb = colorsys.hls_to_rgb(angles / (np.pi * 2), 0.5, 0.5)
    return rgb


@deprecate_arguments({"rho": "state"})
@_optionals.HAS_MATPLOTLIB.require_in_call
@_optionals.HAS_SEABORN.require_in_call
def plot_state_qsphere(
    state,
    figsize=None,
    ax=None,
    show_state_labels=True,
    show_state_phases=False,
    use_degrees=False,
    *,
    rho=None,
    filename=None,
):
    """Plot the qsphere representation of a quantum state.
    Here, the size of the points is proportional to the probability
    of the corresponding term in the state and the color represents
    the phase.

    Args:
        state (Statevector or DensityMatrix or ndarray): an N-qubit quantum state.
        figsize (tuple): Figure size in inches.
        ax (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified a new matplotlib
            Figure will be created and used. Additionally, if specified there
            will be no returned Figure since it is redundant.
        show_state_labels (bool): An optional boolean indicating whether to
            show labels for each basis state.
        show_state_phases (bool): An optional boolean indicating whether to
            show the phase for each basis state.
        use_degrees (bool): An optional boolean indicating whether to use
            radians or degrees for the phase values in the plot.

    Returns:
        Figure: A matplotlib figure instance if the ``ax`` kwarg is not set

    Raises:
        MissingOptionalLibraryError: Requires matplotlib.
        VisualizationError: if input is not a valid N-qubit state.

        QiskitError: Input statevector does not have valid dimensions.

    Examples:
        .. jupyter-execute::

           from qiskit import QuantumCircuit
           from qiskit.quantum_info import Statevector
           from qiskit.visualization import plot_state_qsphere

           qc = QuantumCircuit(2)
           qc.h(0)
           qc.cx(0, 1)

           state = Statevector(qc)
           plot_state_qsphere(state)

        .. jupyter-execute::

           # You can show the phase of each state and use
           # degrees instead of radians

           from qiskit.quantum_info import DensityMatrix
           import numpy as np

           qc = QuantumCircuit(2)
           qc.h([0, 1])
           qc.cz(0,1)
           qc.ry(np.pi/3, 0)
           qc.rx(np.pi/5, 1)
           qc.z(1)

           matrix = DensityMatrix(qc)
           plot_state_qsphere(matrix,
                show_state_phases = True, use_degrees = True)
    """
    from matplotlib import gridspec
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle
    import seaborn as sns
    from scipy import linalg
    from .bloch import Arrow3D

    rho = DensityMatrix(state)
    num = rho.num_qubits
    if num is None:
        raise VisualizationError("Input is not a multi-qubit quantum state.")
    # get the eigenvectors and eigenvalues
    eigvals, eigvecs = linalg.eigh(rho.data)

    if figsize is None:
        figsize = (7, 7)

    if ax is None:
        return_fig = True
        fig = plt.figure(figsize=figsize)
    else:
        return_fig = False
        fig = ax.get_figure()

    gs = gridspec.GridSpec(nrows=3, ncols=3)

    ax = fig.add_subplot(gs[0:3, 0:3], projection="3d")
    ax.axes.set_xlim3d(-1.0, 1.0)
    ax.axes.set_ylim3d(-1.0, 1.0)
    ax.axes.set_zlim3d(-1.0, 1.0)
    ax.axes.grid(False)
    ax.view_init(elev=5, azim=275)

    # Force aspect ratio
    # MPL 3.2 or previous do not have set_box_aspect
    if hasattr(ax.axes, "set_box_aspect"):
        ax.axes.set_box_aspect((1, 1, 1))

    # start the plotting
    # Plot semi-transparent sphere
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 25)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(
        x, y, z, rstride=1, cstride=1, color=plt.rcParams["grid.color"], alpha=0.2, linewidth=0
    )

    # Get rid of the panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the spines
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # traversing the eigvals/vecs backward as sorted low->high
    for idx in range(eigvals.shape[0] - 1, -1, -1):
        if eigvals[idx] > 0.001:
            # get the max eigenvalue
            state = eigvecs[:, idx]
            loc = np.absolute(state).argmax()
            # remove the global phase from max element
            angles = (np.angle(state[loc]) + 2 * np.pi) % (2 * np.pi)
            angleset = np.exp(-1j * angles)
            state = angleset * state

            d = num
            for i in range(2**num):
                # get x,y,z points
                element = bin(i)[2:].zfill(num)
                weight = element.count("1")
                zvalue = -2 * weight / d + 1
                number_of_divisions = n_choose_k(d, weight)
                weight_order = bit_string_index(element)
                angle = (float(weight) / d) * (np.pi * 2) + (
                    weight_order * 2 * (np.pi / number_of_divisions)
                )

                if (weight > d / 2) or (
                    (weight == d / 2) and (weight_order >= number_of_divisions / 2)
                ):
                    angle = np.pi - angle - (2 * np.pi / number_of_divisions)

                xvalue = np.sqrt(1 - zvalue**2) * np.cos(angle)
                yvalue = np.sqrt(1 - zvalue**2) * np.sin(angle)

                # get prob and angle - prob will be shade and angle color
                prob = np.real(np.dot(state[i], state[i].conj()))
                prob = min(prob, 1)  # See https://github.com/Qiskit/qiskit-terra/issues/4666
                colorstate = phase_to_rgb(state[i])

                alfa = 1
                if yvalue >= 0.1:
                    alfa = 1.0 - yvalue

                if not np.isclose(prob, 0) and show_state_labels:
                    rprime = 1.3
                    angle_theta = np.arctan2(np.sqrt(1 - zvalue**2), zvalue)
                    xvalue_text = rprime * np.sin(angle_theta) * np.cos(angle)
                    yvalue_text = rprime * np.sin(angle_theta) * np.sin(angle)
                    zvalue_text = rprime * np.cos(angle_theta)
                    element_text = "$\\vert" + element + "\\rangle$"
                    if show_state_phases:
                        element_angle = (np.angle(state[i]) + (np.pi * 4)) % (np.pi * 2)
                        if use_degrees:
                            element_text += "\n$%.1f^\\circ$" % (element_angle * 180 / np.pi)
                        else:
                            element_angle = pi_check(element_angle, ndigits=3).replace("pi", "\\pi")
                            element_text += "\n$%s$" % (element_angle)
                    ax.text(
                        xvalue_text,
                        yvalue_text,
                        zvalue_text,
                        element_text,
                        ha="center",
                        va="center",
                        size=12,
                    )

                ax.plot(
                    [xvalue],
                    [yvalue],
                    [zvalue],
                    markerfacecolor=colorstate,
                    markeredgecolor=colorstate,
                    marker="o",
                    markersize=np.sqrt(prob) * 30,
                    alpha=alfa,
                )

                a = Arrow3D(
                    [0, xvalue],
                    [0, yvalue],
                    [0, zvalue],
                    mutation_scale=20,
                    alpha=prob,
                    arrowstyle="-",
                    color=colorstate,
                    lw=2,
                )
                ax.add_artist(a)

            # add weight lines
            for weight in range(d + 1):
                theta = np.linspace(-2 * np.pi, 2 * np.pi, 100)
                z = -2 * weight / d + 1
                r = np.sqrt(1 - z**2)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                ax.plot(x, y, z, color=(0.5, 0.5, 0.5), lw=1, ls=":", alpha=0.5)

            # add center point
            ax.plot(
                [0],
                [0],
                [0],
                markerfacecolor=(0.5, 0.5, 0.5),
                markeredgecolor=(0.5, 0.5, 0.5),
                marker="o",
                markersize=3,
                alpha=1,
            )
        else:
            break

    n = 64
    theta = np.ones(n)
    colors = sns.hls_palette(n)

    ax2 = fig.add_subplot(gs[2:, 2:])
    ax2.pie(theta, colors=colors[5 * n // 8 :] + colors[: 5 * n // 8], radius=0.75)
    ax2.add_artist(Circle((0, 0), 0.5, color="white", zorder=1))
    offset = 0.95  # since radius of sphere is one.

    if use_degrees:
        labels = ["Phase\n(Deg)", "0", "90", "180   ", "270"]
    else:
        labels = ["Phase", "$0$", "$\\pi/2$", "$\\pi$", "$3\\pi/2$"]

    ax2.text(0, 0, labels[0], horizontalalignment="center", verticalalignment="center", fontsize=14)
    ax2.text(
        offset, 0, labels[1], horizontalalignment="center", verticalalignment="center", fontsize=14
    )
    ax2.text(
        0, offset, labels[2], horizontalalignment="center", verticalalignment="center", fontsize=14
    )
    ax2.text(
        -offset, 0, labels[3], horizontalalignment="center", verticalalignment="center", fontsize=14
    )
    ax2.text(
        0, -offset, labels[4], horizontalalignment="center", verticalalignment="center", fontsize=14
    )

    if return_fig:
        matplotlib_close_if_inline(fig)
    if filename is None:
        return fig
    else:
        return fig.savefig(filename)


@_optionals.HAS_MATPLOTLIB.require_in_call
def generate_facecolors(x, y, z, dx, dy, dz, color):
    """Generates shaded facecolors for shaded bars.

    This is here to work around a Matplotlib bug
    where alpha does not work in Bar3D.

    Args:
        x (array_like): The x- coordinates of the anchor point of the bars.
        y (array_like): The y- coordinates of the anchor point of the bars.
        z (array_like): The z- coordinates of the anchor point of the bars.
        dx (array_like): Width of bars.
        dy (array_like): Depth of bars.
        dz (array_like): Height of bars.
        color (array_like): sequence of valid color specifications, optional
    Returns:
        list: Shaded colors for bars.
    Raises:
        MissingOptionalLibraryError: If matplotlib is not installed
    """
    import matplotlib.colors as mcolors

    cuboid = np.array(
        [
            # -z
            (
                (0, 0, 0),
                (0, 1, 0),
                (1, 1, 0),
                (1, 0, 0),
            ),
            # +z
            (
                (0, 0, 1),
                (1, 0, 1),
                (1, 1, 1),
                (0, 1, 1),
            ),
            # -y
            (
                (0, 0, 0),
                (1, 0, 0),
                (1, 0, 1),
                (0, 0, 1),
            ),
            # +y
            (
                (0, 1, 0),
                (0, 1, 1),
                (1, 1, 1),
                (1, 1, 0),
            ),
            # -x
            (
                (0, 0, 0),
                (0, 0, 1),
                (0, 1, 1),
                (0, 1, 0),
            ),
            # +x
            (
                (1, 0, 0),
                (1, 1, 0),
                (1, 1, 1),
                (1, 0, 1),
            ),
        ]
    )

    # indexed by [bar, face, vertex, coord]
    polys = np.empty(x.shape + cuboid.shape)
    # handle each coordinate separately
    for i, p, dp in [(0, x, dx), (1, y, dy), (2, z, dz)]:
        p = p[..., np.newaxis, np.newaxis]
        dp = dp[..., np.newaxis, np.newaxis]
        polys[..., i] = p + dp * cuboid[..., i]

    # collapse the first two axes
    polys = polys.reshape((-1,) + polys.shape[2:])

    facecolors = []
    if len(color) == len(x):
        # bar colors specified, need to expand to number of faces
        for c in color:
            facecolors.extend([c] * 6)
    else:
        # a single color specified, or face colors specified explicitly
        facecolors = list(mcolors.to_rgba_array(color))
        if len(facecolors) < len(x):
            facecolors *= 6 * len(x)

    normals = _generate_normals(polys)
    return _shade_colors(facecolors, normals)


def _generate_normals(polygons):
    """Takes a list of polygons and return an array of their normals.

    Normals point towards the viewer for a face with its vertices in
    counterclockwise order, following the right hand rule.
    Uses three points equally spaced around the polygon.
    This normal of course might not make sense for polygons with more than
    three points not lying in a plane, but it's a plausible and fast
    approximation.

    Args:
        polygons (list): list of (M_i, 3) array_like, or (..., M, 3) array_like
            A sequence of polygons to compute normals for, which can have
            varying numbers of vertices. If the polygons all have the same
            number of vertices and array is passed, then the operation will
            be vectorized.
    Returns:
        normals: (..., 3) array_like
            A normal vector estimated for the polygon.
    """
    if isinstance(polygons, np.ndarray):
        # optimization: polygons all have the same number of points, so can
        # vectorize
        n = polygons.shape[-2]
        i1, i2, i3 = 0, n // 3, 2 * n // 3
        v1 = polygons[..., i1, :] - polygons[..., i2, :]
        v2 = polygons[..., i2, :] - polygons[..., i3, :]
    else:
        # The subtraction doesn't vectorize because polygons is jagged.
        v1 = np.empty((len(polygons), 3))
        v2 = np.empty((len(polygons), 3))
        for poly_i, ps in enumerate(polygons):
            n = len(ps)
            i1, i2, i3 = 0, n // 3, 2 * n // 3
            v1[poly_i, :] = ps[i1, :] - ps[i2, :]
            v2[poly_i, :] = ps[i2, :] - ps[i3, :]

    return np.cross(v1, v2)


def _shade_colors(color, normals, lightsource=None):
    """
    Shade *color* using normal vectors given by *normals*.
    *color* can also be an array of the same length as *normals*.
    """
    from matplotlib.colors import Normalize, LightSource
    import matplotlib.colors as mcolors

    if lightsource is None:
        # chosen for backwards-compatibility
        lightsource = LightSource(azdeg=225, altdeg=19.4712)

    def mod(v):
        return np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)

    shade = np.array(
        [np.dot(n / mod(n), lightsource.direction) if mod(n) else np.nan for n in normals]
    )
    mask = ~np.isnan(shade)

    if mask.any():
        norm = Normalize(min(shade[mask]), max(shade[mask]))
        shade[~mask] = min(shade[mask])
        color = mcolors.to_rgba_array(color)
        # shape of color should be (M, 4) (where M is number of faces)
        # shape of shade should be (M,)
        # colors should have final shape of (M, 4)
        alpha = color[:, 3]
        colors = (0.5 + norm(shade)[:, np.newaxis] * 0.5) * color
        colors[:, 3] = alpha
    else:
        colors = np.asanyarray(color).copy()

    return colors


def state_to_latex(
    state: Union[Statevector, DensityMatrix], dims: bool = None, convention: str = "ket", **args
) -> str:
    """Return a Latex representation of a state. Wrapper function
    for `qiskit.visualization.array_to_latex` for convention 'vector'.
    Adds dims if necessary.
    Intended for use within `state_drawer`.

    Args:
        state: State to be drawn
        dims (bool): Whether to display the state's `dims`
        convention (str): Either 'vector' or 'ket'. For 'ket' plot the state in the ket-notation.
                Otherwise plot as a vector
        **args: Arguments to be passed directly to `array_to_latex` for convention 'ket'

    Returns:
        Latex representation of the state
    """
    if dims is None:  # show dims if state is not only qubits
        if set(state.dims()) == {2}:
            dims = False
        else:
            dims = True

    prefix = ""
    suffix = ""
    if dims:
        prefix = "\\begin{align}\n"
        dims_str = state._op_shape.dims_l()
        suffix = f"\\\\\n\\text{{dims={dims_str}}}\n\\end{{align}}"

    operator_shape = state._op_shape
    # we only use the ket convetion for qubit statevectors
    # this means the operator shape should hve no input dimensions and all output dimensions equal to 2
    is_qubit_statevector = len(operator_shape.dims_r()) == 0 and set(operator_shape.dims_l()) == {2}
    if convention == "ket" and is_qubit_statevector:
        latex_str = _state_to_latex_ket(state._data, **args)
    else:
        latex_str = array_to_latex(state._data, source=True, **args)
    return prefix + latex_str + suffix


def num_to_latex_ket(raw_value: complex, first_term: bool, decimals: int = 10) -> Optional[str]:
    """Convert a complex number to latex code suitable for a ket expression

    Args:
        raw_value: Value to convert
        first_term: If True then generate latex code for the first term in an expression
        decimals: Number of decimal places to round to (default: 10).
    Returns:
        String with latex code or None if no term is required
    """
    import sympy  # runtime import

    if raw_value == 0:
        value = 0
        real_value = 0
        imag_value = 0
    else:
        raw_value = np.around(raw_value, decimals=decimals)
        value = sympy.nsimplify(raw_value, constants=(sympy.pi,), rational=False)
        real_value = float(sympy.re(value))
        imag_value = float(sympy.im(value))

    element = ""
    if np.abs(value) > 0:
        latex_element = sympy.latex(value, full_prec=False)
        two_term = real_value != 0 and imag_value != 0
        if isinstance(value, sympy.core.Add):
            # can happen for expressions like 1 + sqrt(2)
            two_term = True
        if two_term:
            if first_term:
                element = f"({latex_element})"
            else:
                element = f"+ ({latex_element})"
        else:
            if first_term:
                if np.isreal(complex(value)) and value > 0:
                    element = latex_element
                else:
                    element = latex_element
                if element == "1":
                    element = ""
                elif element == "-1":
                    element = "-"
            else:

                if imag_value == 0 and real_value > 0:
                    element = "+" + latex_element
                elif real_value == 0 and imag_value > 0:
                    element = "+" + latex_element
                else:
                    element = latex_element
                if element == "+1":
                    element = "+"
                elif element == "-1":
                    element = "-"

        return element
    else:
        return None


def numbers_to_latex_terms(numbers: List[complex], decimals: int = 10) -> List[str]:
    """Convert a list of numbers to latex formatted terms

    The first non-zero term is treated differently. For this term a leading + is suppressed.

    Args:
        numbers: List of numbers to format
        decimals: Number of decimal places to round to (default: 10).
    Returns:
        List of formatted terms
    """
    first_term = True
    terms = []
    for number in numbers:
        term = num_to_latex_ket(number, first_term, decimals)
        if term is not None:
            first_term = False
        terms.append(term)
    return terms


def _state_to_latex_ket(data: List[complex], max_size: int = 12, prefix: str = "") -> str:
    """Convert state vector to latex representation

    Args:
        data: State vector
        max_size: Maximum number of non-zero terms in the expression. If the number of
                 non-zero terms is larger than the max_size, then the representation is truncated.
        prefix: Latex string to be prepended to the latex, intended for labels.

    Returns:
        String with LaTeX representation of the state vector
    """
    num = int(np.log2(len(data)))

    def ket_name(i):
        return bin(i)[2:].zfill(num)

    data = np.around(data, max_size)
    nonzero_indices = np.where(data != 0)[0].tolist()
    if len(nonzero_indices) > max_size:
        nonzero_indices = (
            nonzero_indices[: max_size // 2] + [0] + nonzero_indices[-max_size // 2 + 1 :]
        )
        latex_terms = numbers_to_latex_terms(data[nonzero_indices], max_size)
        nonzero_indices[max_size // 2] = None
    else:
        latex_terms = numbers_to_latex_terms(data[nonzero_indices], max_size)

    latex_str = ""
    for idx, ket_idx in enumerate(nonzero_indices):
        if ket_idx is None:
            latex_str += r" + \ldots "
        else:
            term = latex_terms[idx]
            ket = ket_name(ket_idx)
            latex_str += f"{term} |{ket}\\rangle"
    return prefix + latex_str


class TextMatrix:
    """Text representation of an array, with `__str__` method so it
    displays nicely in Jupyter notebooks"""

    def __init__(self, state, max_size=8, dims=None, prefix="", suffix=""):
        self.state = state
        self.max_size = max_size
        if dims is None:  # show dims if state is not only qubits
            if set(state.dims()) == {2}:
                dims = False
            else:
                dims = True
        self.dims = dims
        self.prefix = prefix
        self.suffix = suffix
        if isinstance(max_size, int):
            self.max_size = max_size
        elif isinstance(state, DensityMatrix):
            # density matrices are square, so threshold for
            # summarization is shortest side squared
            self.max_size = min(max_size) ** 2
        else:
            self.max_size = max_size[0]

    def __str__(self):
        threshold = self.max_size
        data = np.array2string(
            self.state._data, prefix=self.prefix, threshold=threshold, separator=","
        )
        dimstr = ""
        if self.dims:
            data += ",\n"
            dimstr += " " * len(self.prefix)
            dimstr += f"dims={self.state._op_shape.dims_l()}"
        return self.prefix + data + dimstr + self.suffix

    def __repr__(self):
        return self.__str__()


def state_drawer(state, output=None, **drawer_args):
    """Returns a visualization of the state.

    **repr**: ASCII TextMatrix of the state's ``_repr_``.

    **text**: ASCII TextMatrix that can be printed in the console.

    **latex**: An IPython Latex object for displaying in Jupyter Notebooks.

    **latex_source**: Raw, uncompiled ASCII source to generate array using LaTeX.

    **qsphere**: Matplotlib figure, rendering of statevector using `plot_state_qsphere()`.

    **hinton**: Matplotlib figure, rendering of statevector using `plot_state_hinton()`.

    **bloch**: Matplotlib figure, rendering of statevector using `plot_bloch_multivector()`.

    **city**: Matplotlib figure, rendering of statevector using `plot_state_city()`.

    **paulivec**: Matplotlib figure, rendering of statevector using `plot_state_paulivec()`.

    Args:
        output (str): Select the output method to use for drawing the
            circuit. Valid choices are ``text``, ``latex``, ``latex_source``,
            ``qsphere``, ``hinton``, ``bloch``, ``city`` or ``paulivec``.
            Default is `'text`'.
        drawer_args: Arguments to be passed to the relevant drawer. For
            'latex' and 'latex_source' see ``array_to_latex``

    Returns:
        :class:`matplotlib.figure` or :class:`str` or
        :class:`TextMatrix` or :class:`IPython.display.Latex`:
        Drawing of the state.

    Raises:
        MissingOptionalLibraryError: when `output` is `latex` and IPython is not installed.
        ValueError: when `output` is not a valid selection.
    """
    config = user_config.get_config()
    # Get default 'output' from config file else use 'repr'
    default_output = "repr"
    if output is None:
        if config:
            default_output = config.get("state_drawer", "repr")
        output = default_output
    output = output.lower()

    # Choose drawing backend:
    drawers = {
        "text": TextMatrix,
        "latex_source": state_to_latex,
        "qsphere": plot_state_qsphere,
        "hinton": plot_state_hinton,
        "bloch": plot_bloch_multivector,
        "city": plot_state_city,
        "paulivec": plot_state_paulivec,
    }
    if output == "latex":
        _optionals.HAS_IPYTHON.require_now("state_drawer")
        from IPython.display import Latex

        draw_func = drawers["latex_source"]
        return Latex(f"$${draw_func(state, **drawer_args)}$$")

    if output == "repr":
        return state.__repr__()

    try:
        draw_func = drawers[output]
        return draw_func(state, **drawer_args)
    except KeyError as err:
        raise ValueError(
            """'{}' is not a valid option for drawing {} objects. Please choose from:
            'text', 'latex', 'latex_source', 'qsphere', 'hinton',
            'bloch', 'city' or 'paulivec'.""".format(
                output, type(state).__name__
            )
        ) from err


def _bloch_multivector_data(state):
    """Return list of Bloch vectors for each qubit

    Args:
        state (DensityMatrix or Statevector): an N-qubit state.

    Returns:
        list: list of Bloch vectors (x, y, z) for each qubit.

    Raises:
        VisualizationError: if input is not an N-qubit state.
    """
    rho = DensityMatrix(state)
    num = rho.num_qubits
    if num is None:
        raise VisualizationError("Input is not a multi-qubit quantum state.")
    pauli_singles = PauliList(["X", "Y", "Z"])
    bloch_data = []
    for i in range(num):
        if num > 1:
            paulis = PauliList.from_symplectic(
                np.zeros((3, (num - 1)), dtype=bool), np.zeros((3, (num - 1)), dtype=bool)
            ).insert(i, pauli_singles, qubit=True)
        else:
            paulis = pauli_singles
        bloch_state = [np.real(np.trace(np.dot(mat, rho.data))) for mat in paulis.matrix_iter()]
        bloch_data.append(bloch_state)
    return bloch_data


def _paulivec_data(state):
    """Return paulivec data for plotting.

    Args:
        state (DensityMatrix or Statevector): an N-qubit state.

    Returns:
        tuple: (labels, values) for Pauli vector.

    Raises:
        VisualizationError: if input is not an N-qubit state.
    """
    rho = SparsePauliOp.from_operator(DensityMatrix(state))
    if rho.num_qubits is None:
        raise VisualizationError("Input is not a multi-qubit quantum state.")
    return rho.paulis.to_labels(), np.real(rho.coeffs)
