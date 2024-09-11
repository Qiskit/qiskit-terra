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

"""Faulty fake backends for testing"""

from qiskit.providers.models.backendproperties import BackendProperties
from qiskit.providers.fake_provider import Fake7QPulseV1


class Fake7QV1FaultyQ1(Fake7QPulseV1):
    """A fake 5 qubit backend, with a faulty q1
    0 ↔ (1) ↔ 3 ↔ 4
         ↕
         2
    """

    def properties(self):
        """Returns a snapshot of device properties as recorded on 8/30/19.
        Sets the qubit 1 as non-operational.
        """
        props = super().properties().to_dict()
        props["qubits"][1].append(
            {"date": "2000-01-01 00:00:00Z", "name": "operational", "unit": "", "value": 0}
        )
        return BackendProperties.from_dict(props)


class Fake7QV1MissingQ1Property(Fake7QPulseV1):
    """A fake 7 qubit backend, with a missing T1 property in q1."""

    def properties(self):
        """Returns a snapshot of device properties as recorded on 8/30/19.
        Remove property from qubit 1.
        """
        props = super().properties().to_dict()
        props["qubits"][1] = filter(lambda q: q["name"] != "T1", props["qubits"][1])
        return BackendProperties.from_dict(props)


class Fake7QV1FaultyCX01CX10(Fake7QPulseV1):
    """A fake 5 qubit backend, with faulty CX(Q1, Q0) and CX(Q0, Q1)
    0 (↔) 1 ↔ 3 ↔ 4
          ↕
          2
    """

    def properties(self):
        """Returns a snapshot of device properties as recorded on 8/30/19.
        Sets the gate CX(Q0, Q1) (and symmetric) as non-operational.
        """
        props = super().properties().to_dict()
        for gate in props["gates"]:
            if gate["gate"] == "cx" and set(gate["qubits"]) == {0, 1}:
                gate["parameters"].append(
                    {"date": "2000-01-01 00:00:00Z", "name": "operational", "unit": "", "value": 0}
                )
        return BackendProperties.from_dict(props)


class Fake7QV1FaultyCX13CX31(Fake7QPulseV1):
    """A fake 5 qubit backend, with faulty CX(Q1, Q3) and CX(Q3, Q1)
    0 ↔ 1 (↔) 3 ↔ 4
        ↕
        2
    """

    def properties(self):
        """Returns a snapshot of device properties as recorded on 8/30/19.
        Sets the gate CX(Q1, Q3) (and symmetric) as non-operational.
        """
        props = super().properties().to_dict()
        for gate in props["gates"]:
            if gate["gate"] == "cx" and set(gate["qubits"]) == {3, 1}:
                gate["parameters"].append(
                    {"date": "2000-01-01 00:00:00Z", "name": "operational", "unit": "", "value": 0}
                )
        return BackendProperties.from_dict(props)


class Fake7QV1FaultyCX13(Fake7QPulseV1):
    """A fake 5 qubit backend, with faulty CX(Q1, Q3), but valid CX(Q3, Q1)
    0 ↔ 1 <- 3 ↔ 4
        ↕
        2
    """

    def properties(self):
        """Returns a snapshot of device properties as recorded on 8/30/19.
        Sets the gate CX(Q1, Q3) as non-operational.
        """
        props = super().properties().to_dict()
        for gate in props["gates"]:
            if gate["gate"] == "cx" and gate["qubits"] == [1, 3]:
                gate["parameters"].append(
                    {"date": "2000-01-01 00:00:00Z", "name": "operational", "unit": "", "value": 0}
                )
        return BackendProperties.from_dict(props)
