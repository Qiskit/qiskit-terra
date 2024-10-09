# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Matplotlib circuit visualization style."""

from __future__ import annotations

import json
import os
from typing import Any
from pathlib import Path
from warnings import warn
import qiskit.visualization.exceptions as exceptions


class StyleDict(dict):
    """A dictionary for graphviz styles.

    Defines additional abbreviations for key accesses, such as allowing
    ``"ec"`` instead of writing ``"edgecolor"``.
    """

    VALID_FIELDS = {
        "name",
        "fontsize",
        "bgcolor",
        "dpi",
        "pad",
        "nodecolor",
        "inputnodecolor",
        "inputnodefontcolor",
        "outputnodecolor",
        "outputnodefontcolor",
        "opnodecolor",
        "opnodefontcolor",
        "edgecolor",
        "qubitedgecolor",
        "clbitedgecolor",
    }

    ABBREVIATIONS = {
        "nc": "nodecolor",
        "ic": "inputnodecolor",
        "if": "inputnodefontcolor",
        "oc": "outputnodecolor",
        "of": "outputnodefontcolor",
        "opc": "opnodecolor",
        "opf": "opnodefontcolor",
        "ec": "edgecolor",
        "qec": "qubitedgecolor",
        "clc": "clbitedgecolor",
    }

    def __setitem__(self, key: Any, value: Any) -> None:
        # allow using field abbreviations
        if key in self.ABBREVIATIONS:
            key = self.ABBREVIATIONS[key]

        if key not in self.VALID_FIELDS:
            warn(
                f"style option ({key}) is not supported",
                UserWarning,
                2,
            )
        return super().__setitem__(key, value)

    def __getitem__(self, key: Any) -> Any:
        # allow using field abbreviations
        if key in self.ABBREVIATIONS:
            key = self.ABBREVIATIONS[key]

        return super().__getitem__(key)

    def update(self, other):
        super().update((key, value) for key, value in other.items())


class DefaultStyle:
    """Creates a Default Style dictionary

    The style dict contains numerous options that define the style of the
    output circuit visualization. The style dict is used by the `mpl` or
    `latex` output. The options available in the style dict are defined below:

    Attributes:
        name (str): The name of the style.
        fontsize (str): The font size to use for text.
        bgcolor (str): The color name to use for the background ('red', 'green', etc.).
        nodecolor (str): The color to use for all nodes.
        dpi (int): The DPI to use for the output image.
        pad (int): A number to adjust padding around output
            graph.
        inputnodecolor (str): The color to use for incoming wire nodes. Overrides
            nodecolor for those nodes.
        inputnodefontcolor (str): The font color to use for incoming wire nodes.
            Overrides nodecolor for those nodes.
        outputnodecolor (str): The color to use for output wire nodes. Overrides
            nodecolor for those nodes.
        outputnodefontcolor (str): The font color to use for output wire nodes.
            Overrides nodecolor for those nodes.
        opnodecolor (str): The color to use for Instruction nodes. Overrides
            nodecolor for those nodes.
        opnodefontcolor (str): The font color to use for Instruction nodes.
            Overrides nodecolor for those nodes.

        qubitedgecolor (str): The edge color for qubits. Overrides edgecolor for these edges.
        clbitedgecolor (str): The edge color for clbits. Overrides edgecolor for these edges.
    """

    def __init__(self):
        default_style_dict = "color.json"
        path = Path(__file__).parent / "styles" / default_style_dict

        with open(path, "r") as infile:
            default_style = json.load(infile)

        # set shortcuts, such as "ec" for "edgecolor"
        self.style = StyleDict(**default_style)


def load_style(style: dict | str = "color") -> StyleDict:
    """Utility function to load style from json files.

    Args:
        style: Depending on the type, this acts differently:

            * If a string, it can specify a supported style name (such
              as "iqp" or "clifford"). It can also specify the name of
              a custom color scheme stored as JSON file. This JSON file
              _must_ specify a complete set of colors.
            * If a dictionary, it may specify the style name via a
              ``{"name": "<desired style>"}`` entry. If this is not given,
              the default style will be used. The remaining entries in the
              dictionary can be used to override certain specs.
              E.g. ``{"name": "iqp", "ec": "#FF0000"}`` will use the ``"iqp"``
              color scheme but set the edgecolor to red.

    Returns:
        A tuple containing the style as dictionary and the default font ratio.
    """

    # determine the style name which could also be inside a dictionary, like
    # style={"name": "clifford", <other settings...>}
    if isinstance(style, dict):
        style_name = style.get("name", "color")
    elif isinstance(style, str):
        if style.endswith(".json"):
            style_name = style[:-5]
        else:
            style_name = style
    else:
        raise exceptions.VisualizationError("Invalid style {style}")

    if style_name in ["color"]:
        current_style = DefaultStyle().style
    else:
        # Search for file in 'styles' dir, and then the current directory
        style_name = style_name + ".json"
        style_paths = []

        default_path = Path(__file__).parent / "styles" / style_name
        style_paths.append(default_path)

        # check current directory
        cwd_path = Path("") / style_name
        style_paths.append(cwd_path)

        for path in style_paths:
            # expand ~ to the user directory and check if the file exists
            exp_user = path.expanduser()
            if os.path.isfile(exp_user):
                try:
                    with open(exp_user) as infile:
                        json_style = json.load(infile)

                    current_style = StyleDict(json_style)
                    break
                except json.JSONDecodeError as err:
                    warn(
                        f"Could not decode JSON in file '{path}': {str(err)}. "
                        "Will use default style.",
                        UserWarning,
                        2,
                    )
                    break
                except (OSError, FileNotFoundError):
                    warn(
                        f"Error loading JSON file '{path}'. Will use default style.",
                        UserWarning,
                        2,
                    )
                    break
        else:
            raise exceptions.VisualizationError(f"Invalid style {style}")

    # if the style is a dictionary, update the defaults with the new values
    # this _needs_ to happen after loading by name to cover cases like
    #   style = {"name": "bw", "edgecolor": "#FF0000"}
    if isinstance(style, dict):
        current_style.update(style)

    return current_style
