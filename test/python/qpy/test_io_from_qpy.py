# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test cases for qpy loading and saving."""

import io

from qiskit import QuantumCircuit
from qiskit.qpy import dump, load


def test_io():
    """Test QPy serialization and de-serialization."""

    circuit = QuantumCircuit(2)
    circuit.ms(1.23, [0, 1])

    buf = io.BytesIO()
    dump(circuit, buf)

    try:
        _ = load(io.BytesIO(buf.getvalue()))
    except Exception as exception:
        raise IOError("Loading to and from qpy failed.") from exception
