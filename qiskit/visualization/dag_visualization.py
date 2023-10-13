# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""
Visualization function for DAG circuit representation.
"""
from rustworkx.visualization import graphviz_draw

from qiskit.dagcircuit.dagnode import DAGOpNode, DAGInNode, DAGOutNode
from qiskit.circuit import Qubit
from qiskit.utils import optionals as _optionals
from qiskit.exceptions import InvalidFileError
from .exceptions import VisualizationError


@_optionals.HAS_GRAPHVIZ.require_in_call
def dag_drawer(dag, scale=0.7, filename=None, style="color", show_node_id=False):
    """Plot the directed acyclic graph (dag) to represent operation dependencies
    in a quantum circuit.

    This function calls the :func:`~rustworkx.visualization.graphviz_draw` function from the
    ``rustworkx`` package to draw the DAG.

    Args:
        dag (DAGCircuit): The dag to draw.
        scale (float): scaling factor
        filename (str): file path to save image to (format inferred from name)
        style (str): 'plain': B&W graph
                     'color' (default): color input/output/op nodes
        show_node_id (bool): show the graph node_id for each node

    Returns:
        PIL.Image: if in Jupyter notebook and not saving to file,
            otherwise None.

    Raises:
        VisualizationError: when style is not recognized.
        InvalidFileError: when filename provided is not valid

    Example:
        .. plot::
           :include-source:

            from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
            from qiskit.dagcircuit import DAGCircuit
            from qiskit.converters import circuit_to_dag
            from qiskit.visualization import dag_drawer

            q = QuantumRegister(3, 'q')
            c = ClassicalRegister(3, 'c')
            circ = QuantumCircuit(q, c)
            circ.h(q[0])
            circ.cx(q[0], q[1])
            circ.measure(q[0], c[0])
            circ.rz(0.5, q[1]).c_if(c, 2)

            dag = circuit_to_dag(circ)
            dag_drawer(dag)
    """

    # NOTE: use type str checking to avoid potential cyclical import
    # the two tradeoffs ere that it will not handle subclasses and it is
    # slower (which doesn't matter for a visualization function)
    type_str = str(type(dag))
    if "DAGDependency" in type_str:
        graph_attrs = {"dpi": str(100 * scale)}

        def node_attr_func(node):
            if style == "plain":
                return {}
            if style == "color":
                n = {}
                n["label"] = str(node.node_id) + ": " + str(node.name)
                if node.name == "measure":
                    n["color"] = "blue"
                    n["style"] = "filled"
                    n["fillcolor"] = "lightblue"
                if node.name == "barrier":
                    n["color"] = "black"
                    n["style"] = "filled"
                    n["fillcolor"] = "green"
                if getattr(node.op, "_directive", False):
                    n["color"] = "black"
                    n["style"] = "filled"
                    n["fillcolor"] = "red"
                if getattr(node.op, "condition", None):
                    n["label"] = str(node.node_id) + ": " + str(node.name) + " (conditional)"
                    n["color"] = "black"
                    n["style"] = "filled"
                    n["fillcolor"] = "lightgreen"
                return n
            else:
                raise VisualizationError("Unrecognized style %s for the dag_drawer." % style)

        edge_attr_func = None

    else:
        register_bit_labels = {
            bit: f"{reg.name}[{idx}]"
            for reg in list(dag.qregs.values()) + list(dag.cregs.values())
            for (idx, bit) in enumerate(reg)
        }

        graph_attrs = {"dpi": str(100 * scale)}

        def node_attr_func(node):
            if style == "plain":
                return {}
            if style == "color":
                n = {}
                if isinstance(node, DAGOpNode):
                    n["label"] = node.name
                    n["color"] = "blue"
                    n["style"] = "filled"
                    n["fillcolor"] = "lightblue"
                if isinstance(node, DAGInNode):
                    if isinstance(node.wire, Qubit):
                        label = register_bit_labels.get(
                            node.wire, f"q_{dag.find_bit(node.wire).index}"
                        )
                    else:
                        label = register_bit_labels.get(
                            node.wire, f"c_{dag.find_bit(node.wire).index}"
                        )
                    n["label"] = label
                    n["color"] = "black"
                    n["style"] = "filled"
                    n["fillcolor"] = "green"
                if isinstance(node, DAGOutNode):
                    if isinstance(node.wire, Qubit):
                        label = register_bit_labels.get(
                            node.wire, f"q[{dag.find_bit(node.wire).index}]"
                        )
                    else:
                        label = register_bit_labels.get(
                            node.wire, f"c[{dag.find_bit(node.wire).index}]"
                        )
                    n["label"] = label
                    n["color"] = "black"
                    n["style"] = "filled"
                    n["fillcolor"] = "red"
                if show_node_id:
                    n["label"] = f"{node._node_id}: {n['label']}"
                return n
            else:
                raise VisualizationError("Invalid style %s" % style)

        def edge_attr_func(edge):
            e = {}
            if isinstance(edge, Qubit):
                label = register_bit_labels.get(edge, f"q_{dag.find_bit(edge).index}")
            else:
                label = register_bit_labels.get(edge, f"c_{dag.find_bit(edge).index}")
            e["label"] = label
            return e

    image_type = None
    if filename:
        if "." not in filename:
            raise InvalidFileError("Parameter 'filename' must be in format 'name.extension'")
        image_type = filename.split(".")[-1]
    return graphviz_draw(
        dag._multi_graph,
        node_attr_func,
        edge_attr_func,
        graph_attrs,
        filename,
        image_type,
    )
