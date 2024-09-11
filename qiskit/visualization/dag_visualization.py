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

import io
import subprocess

from rustworkx.visualization import graphviz_draw

from qiskit.dagcircuit.dagnode import DAGOpNode, DAGInNode, DAGOutNode
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.circuit import Qubit, Clbit, ClassicalRegister
from qiskit.circuit.classical import expr
from qiskit.converters import dagdependency_to_circuit
from qiskit.utils import optionals as _optionals
from qiskit.exceptions import InvalidFileError
from .exceptions import VisualizationError


IMAGE_TYPES = {
    "canon",
    "cmap",
    "cmapx",
    "cmapx_np",
    "dia",
    "dot",
    "fig",
    "gd",
    "gd2",
    "gif",
    "hpgl",
    "imap",
    "imap_np",
    "ismap",
    "jpe",
    "jpeg",
    "jpg",
    "mif",
    "mp",
    "pcl",
    "pdf",
    "pic",
    "plain",
    "plain-ext",
    "png",
    "ps",
    "ps2",
    "svg",
    "svgz",
    "vml",
    "vmlz",
    "vrml",
    "vtx",
    "wbmp",
    "xdor",
    "xlib",
}


@_optionals.HAS_GRAPHVIZ.require_in_call
@_optionals.HAS_PIL.require_in_call
def dag_drawer(dag, scale=0.7, filename=None, style="color"):
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

    Returns:
        PIL.Image: if in Jupyter notebook and not saving to file,
            otherwise None.

    Raises:
        VisualizationError: when style is not recognized.
        InvalidFileError: when filename provided is not valid
        ValueError: If the file extension for ``filename`` is not an image
            type supported by Graphviz.

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
    from PIL import Image

    # NOTE: use type str checking to avoid potential cyclical import
    # the two tradeoffs ere that it will not handle subclasses and it is
    # slower (which doesn't matter for a visualization function)
    type_str = str(type(dag))
    register_bit_labels = {
        bit: f"{reg.name}[{idx}]"
        for reg in list(dag.qregs.values()) + list(dag.cregs.values())
        for (idx, bit) in enumerate(reg)
    }
    if "DAGDependency" in type_str:
        # pylint: disable=cyclic-import
        from qiskit.visualization.circuit._utils import get_bit_reg_index

        qubit_indices = {bit: index for index, bit in enumerate(dag.qubits)}
        clbit_indices = {bit: index for index, bit in enumerate(dag.clbits)}
        graph_attrs = {"dpi": str(100 * scale)}
        dag_dep_circ = dagdependency_to_circuit(dag)

        def node_attr_func(node):
            if "DAGDependencyV2" in type_str:
                nid_str = str(node._node_id)
            else:
                nid_str = str(node.node_id)
            if style == "plain":
                return {}
            if style == "color":
                n = {}
                args = []
                for count, arg in enumerate(node.qargs + node.cargs):
                    if count > 4:
                        args.append("...")
                        break
                    if isinstance(arg, Qubit):
                        f_str = f"q_{qubit_indices[arg]}"
                    elif isinstance(arg, Clbit):
                        f_str = f"c_{clbit_indices[arg]}"
                    else:
                        f_str = f"{arg.index}"
                    arg_str = register_bit_labels.get(arg, f_str)
                    args.append(arg_str)

                n["color"] = "black"
                n["label"] = (
                    nid_str + ": " + str(node.name) + " (" + str(args)[1:-1].replace("'", "") + ")"
                )
                if node.name == "barrier":
                    n["style"] = "filled"
                    n["fillcolor"] = "grey"
                elif getattr(node.op, "_directive", False):
                    n["style"] = "filled"
                    n["fillcolor"] = "red"
                elif getattr(node.op, "condition", None):
                    condition = node.op.condition
                    if isinstance(condition, expr.Expr):
                        cond_txt = " (cond: [Expr]) ("
                    elif isinstance(condition[0], ClassicalRegister):
                        cond_txt = f" (cond: {condition[0].name}, {int(condition[1])}) ("
                    else:
                        register, bit_index, reg_index = get_bit_reg_index(
                            dag_dep_circ, condition[0]
                        )
                        if register is not None:
                            cond_txt = (
                                f" (cond: {register.name}[{reg_index}], {int(condition[1])}) ("
                            )
                        else:
                            cond_txt = f" (cond: {bit_index}, {int(condition[1])}) ("
                    n["style"] = "filled"
                    n["fillcolor"] = "green"
                    n["label"] = (
                        nid_str
                        + ": "
                        + str(node.name)
                        + cond_txt
                        + str(args)[1:-1].replace("'", "")
                        + ")"
                    )
                elif node.name != "measure":  # measure is unfilled
                    n["style"] = "filled"
                    n["fillcolor"] = "lightblue"
                return n
            else:
                raise VisualizationError(f"Unrecognized style {style} for the dag_drawer.")

        edge_attr_func = None

    else:
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
                    elif isinstance(node.wire, Clbit):
                        label = register_bit_labels.get(
                            node.wire, f"c_{dag.find_bit(node.wire).index}"
                        )
                    else:
                        label = str(node.wire.name)

                    n["label"] = label
                    n["color"] = "black"
                    n["style"] = "filled"
                    n["fillcolor"] = "green"
                if isinstance(node, DAGOutNode):
                    if isinstance(node.wire, Qubit):
                        label = register_bit_labels.get(
                            node.wire, f"q[{dag.find_bit(node.wire).index}]"
                        )
                    elif isinstance(node.wire, Clbit):
                        label = register_bit_labels.get(
                            node.wire, f"c[{dag.find_bit(node.wire).index}]"
                        )
                    else:
                        label = str(node.wire.name)
                    n["label"] = label
                    n["color"] = "black"
                    n["style"] = "filled"
                    n["fillcolor"] = "red"
                return n
            else:
                raise VisualizationError(f"Invalid style {style}")

        def edge_attr_func(edge):
            e = {}
            if isinstance(edge, Qubit):
                label = register_bit_labels.get(edge, f"q_{dag.find_bit(edge).index}")
            elif isinstance(edge, Clbit):
                label = register_bit_labels.get(edge, f"c_{dag.find_bit(edge).index}")
            else:
                label = str(edge.name)
            e["label"] = label
            return e

    image_type = "png"
    if filename:
        if "." not in filename:
            raise InvalidFileError("Parameter 'filename' must be in format 'name.extension'")
        image_type = filename.split(".")[-1]
        if image_type not in IMAGE_TYPES:
            raise ValueError(
                "The specified value for the image_type argument, "
                f"'{image_type}' is not a valid choice. It must be one of: "
                f"{IMAGE_TYPES}"
            )

    if isinstance(dag, DAGCircuit):
        dot_str = dag._to_dot(
            graph_attrs,
            node_attr_func,
            edge_attr_func,
        )

        prog = "dot"
        if not filename:
            dot_result = subprocess.run(
                [prog, "-T", image_type],
                input=dot_str.encode("utf-8"),
                capture_output=True,
                encoding=None,
                check=True,
                text=False,
            )
            dot_bytes_image = io.BytesIO(dot_result.stdout)
            image = Image.open(dot_bytes_image)
            return image
        else:
            subprocess.run(
                [prog, "-T", image_type, "-o", filename],
                input=dot_str,
                check=True,
                encoding="utf8",
                text=True,
            )
            return None
    else:
        return graphviz_draw(
            dag._multi_graph,
            node_attr_func,
            edge_attr_func,
            graph_attrs,
            filename,
            image_type,
        )
