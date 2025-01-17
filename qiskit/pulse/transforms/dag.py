# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""A collection of functions to convert ScheduleBlock to DAG representation."""
from __future__ import annotations

import typing

import rustworkx as rx


from qiskit.pulse.channels import Channel
from qiskit.pulse.exceptions import UnassignedReferenceError

if typing.TYPE_CHECKING:
    from qiskit.pulse import ScheduleBlock  # pylint: disable=cyclic-import


def block_to_dag(block: ScheduleBlock) -> rx.PyDAG:
    """Convert schedule block instruction into DAG.

    ``ScheduleBlock`` can be represented as a DAG as needed.
    For example, equality of two programs are efficiently checked on DAG representation.

    .. plot::
       :include-source:
       :nofigs:
       :context: reset

        from qiskit import pulse

        my_gaussian0 = pulse.Gaussian(100, 0.5, 20)
        my_gaussian1 = pulse.Gaussian(100, 0.3, 10)

        with pulse.build() as sched1:
            with pulse.align_left():
                pulse.play(my_gaussian0, pulse.DriveChannel(0))
                pulse.shift_phase(1.57, pulse.DriveChannel(2))
                pulse.play(my_gaussian1, pulse.DriveChannel(1))

        with pulse.build() as sched2:
            with pulse.align_left():
                pulse.shift_phase(1.57, pulse.DriveChannel(2))
                pulse.play(my_gaussian1, pulse.DriveChannel(1))
                pulse.play(my_gaussian0, pulse.DriveChannel(0))

    Here the ``sched1 `` and ``sched2`` are different implementations of the same program,
    but it is difficult to confirm on the list representation.

    Another example is instruction optimization.

    .. plot::
       :include-source:
       :nofigs:
       :context:

        from qiskit import pulse

        with pulse.build() as sched:
            with pulse.align_left():
                pulse.shift_phase(1.57, pulse.DriveChannel(1))
                pulse.play(my_gaussian0, pulse.DriveChannel(0))
                pulse.shift_phase(-1.57, pulse.DriveChannel(1))

    In above program two ``shift_phase`` instructions can be cancelled out because
    they are consecutive on the same drive channel.
    This can be easily found on the DAG representation.

    Args:
        block ("ScheduleBlock"): A schedule block to be converted.

    Returns:
        Instructions in DAG representation.

    Raises:
        PulseError: When the context is invalid subclass.
    """
    if block.alignment_context.is_sequential:
        return _sequential_allocation(block)
    return _parallel_allocation(block)


def _sequential_allocation(block) -> rx.PyDAG:
    """A helper function to create a DAG of a sequential alignment context."""
    dag = rx.PyDAG()

    edges: list[tuple[int, int]] = []
    prev_id = None
    for elm in block.blocks:
        node_id = dag.add_node(elm)
        if dag.num_nodes() > 1:
            edges.append((prev_id, node_id))
        prev_id = node_id
    dag.add_edges_from_no_data(edges)
    return dag


def _parallel_allocation(block) -> rx.PyDAG:
    """A helper function to create a DAG of a parallel alignment context."""
    dag = rx.PyDAG()

    slots: dict[Channel, int] = {}
    edges: set[tuple[int, int]] = set()
    prev_reference = None
    for elm in block.blocks:
        node_id = dag.add_node(elm)
        try:
            for chan in elm.channels:
                prev_id = slots.pop(chan, prev_reference)
                if prev_id is not None:
                    edges.add((prev_id, node_id))
                slots[chan] = node_id
        except UnassignedReferenceError:
            # Broadcasting channels because the reference's channels are unknown.
            for chan, prev_id in slots.copy().items():
                edges.add((prev_id, node_id))
                slots[chan] = node_id
            prev_reference = node_id
    dag.add_edges_from_no_data(list(edges))
    return dag
