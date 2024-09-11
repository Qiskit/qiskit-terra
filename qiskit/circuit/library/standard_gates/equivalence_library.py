# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Standard gates."""

from __future__ import annotations

from math import pi

from qiskit.circuit import (
    EquivalenceLibrary,
    Parameter,
    QuantumCircuit,
    QuantumRegister,
    Gate,
    Qubit,
    Clbit,
)

from . import (
    HGate,
    CHGate,
    PhaseGate,
    CPhaseGate,
    RGate,
    RCCXGate,
    RXGate,
    CRXGate,
    RXXGate,
    RYGate,
    CRYGate,
    RZGate,
    CRZGate,
    RZZGate,
    RZXGate,
    SGate,
    SdgGate,
    CSGate,
    CSdgGate,
    SwapGate,
    CSwapGate,
    iSwapGate,
    SXGate,
    SXdgGate,
    CSXGate,
    DCXGate,
    TGate,
    TdgGate,
    UGate,
    CUGate,
    U1Gate,
    CU1Gate,
    U2Gate,
    U3Gate,
    CU3Gate,
    XGate,
    CXGate,
    CCXGate,
    YGate,
    CYGate,
    RYYGate,
    ECRGate,
    ZGate,
    CZGate,
    IGate,
    CCZGate,
    XXPlusYYGate,
    XXMinusYYGate,
)


_sel = StandardEquivalenceLibrary = EquivalenceLibrary()


def _cnot_rxx_decompose(plus_ry: bool = True, plus_rxx: bool = True):
    """Decomposition of CNOT gate.

    NOTE: this differs to CNOT by a global phase.
    The matrix returned is given by exp(1j * pi/4) * CNOT

    Args:
        plus_ry (bool): positive initial RY rotation
        plus_rxx (bool): positive RXX rotation.

    Returns:
        QuantumCircuit: The decomposed circuit for CNOT gate (up to
        global phase).
    """
    # Convert boolean args to +/- 1 signs
    if plus_ry:
        sgn_ry = 1
    else:
        sgn_ry = -1
    if plus_rxx:
        sgn_rxx = 1
    else:
        sgn_rxx = -1
    circuit = QuantumCircuit(2, global_phase=-sgn_ry * sgn_rxx * pi / 4)
    circuit.append(RYGate(sgn_ry * pi / 2), [0])
    circuit.append(RXXGate(sgn_rxx * pi / 2), [0, 1])
    circuit.append(RXGate(-sgn_rxx * pi / 2), [0])
    circuit.append(RXGate(-sgn_rxx * sgn_ry * pi / 2), [1])
    circuit.append(RYGate(-sgn_ry * pi / 2), [0])
    return circuit


# Import existing gate definitions

# HGate
#
#    ┌───┐        ┌─────────┐
# q: ┤ H ├  ≡  q: ┤ U2(0,π) ├
#    └───┘        └─────────┘
q = QuantumRegister(1, "q")
def_h = QuantumCircuit(q)
def_h.append(U2Gate(0, pi), [q[0]], [])
_sel.add_equivalence(HGate(), def_h)

# CHGate
#
# q_0: ──■──     q_0: ─────────────────■─────────────────────
#      ┌─┴─┐  ≡       ┌───┐┌───┐┌───┐┌─┴─┐┌─────┐┌───┐┌─────┐
# q_1: ┤ H ├     q_1: ┤ S ├┤ H ├┤ T ├┤ X ├┤ Tdg ├┤ H ├┤ Sdg ├
#      └───┘          └───┘└───┘└───┘└───┘└─────┘└───┘└─────┘
q = QuantumRegister(2, "q")
def_ch = QuantumCircuit(q)
for inst, qargs, cargs in [
    (SGate(), [q[1]], []),
    (HGate(), [q[1]], []),
    (TGate(), [q[1]], []),
    (CXGate(), [q[0], q[1]], []),
    (TdgGate(), [q[1]], []),
    (HGate(), [q[1]], []),
    (SdgGate(), [q[1]], []),
]:
    def_ch.append(inst, qargs, cargs)
_sel.add_equivalence(CHGate(), def_ch)

# PhaseGate
#
#    ┌──────┐        ┌───────┐
# q: ┤ P(ϴ) ├  ≡  q: ┤ U1(ϴ) ├
#    └──────┘        └───────┘
q = QuantumRegister(1, "q")
theta = Parameter("theta")
phase_to_u1 = QuantumCircuit(q)
phase_to_u1.append(U1Gate(theta), [0])
_sel.add_equivalence(PhaseGate(theta), phase_to_u1)

q = QuantumRegister(1, "q")
theta = Parameter("theta")
phase_to_u = QuantumCircuit(q)
phase_to_u.u(0, 0, theta, 0)
_sel.add_equivalence(PhaseGate(theta), phase_to_u)

# CPhaseGate
#                      ┌────────┐
# q_0: ─■────     q_0: ┤ P(ϴ/2) ├──■───────────────■────────────
#       │P(ϴ)  ≡       └────────┘┌─┴─┐┌─────────┐┌─┴─┐┌────────┐
# q_1: ─■────     q_1: ──────────┤ X ├┤ P(-ϴ/2) ├┤ X ├┤ P(ϴ/2) ├
#                                └───┘└─────────┘└───┘└────────┘
q = QuantumRegister(2, "q")
theta = Parameter("theta")
def_cphase = QuantumCircuit(q)
def_cphase.p(theta / 2, 0)
def_cphase.cx(0, 1)
def_cphase.p(-theta / 2, 1)
def_cphase.cx(0, 1)
def_cphase.p(theta / 2, 1)
_sel.add_equivalence(CPhaseGate(theta), def_cphase)

# CPhaseGate
#
# q_0: ─■────     q_0: ─■────
#       │P(ϴ)  ≡        │U1(ϴ)
# q_1: ─■────     q_1: ─■────
q = QuantumRegister(2, "q")
theta = Parameter("theta")
cphase_to_cu1 = QuantumCircuit(q)
cphase_to_cu1.append(CU1Gate(theta), [0, 1])
_sel.add_equivalence(CPhaseGate(theta), cphase_to_cu1)

# CPhaseGate
#
#                  global phase: ϴ/4
#                                  ┌─────────┐
#  q_0: ─■────     q_0: ─■─────────┤ Rz(ϴ/2) ├
#        │P(ϴ)  ≡        │ZZ(-ϴ/2) ├─────────┤
#  q_1: ─■────     q_1: ─■─────────┤ Rz(ϴ/2) ├
#                                  └─────────┘
theta = Parameter("theta")
cphase_to_rzz = QuantumCircuit(2, global_phase=theta / 4)
cphase_to_rzz.rzz(-theta / 2, 0, 1)
cphase_to_rzz.rz(theta / 2, 0)
cphase_to_rzz.rz(theta / 2, 1)
_sel.add_equivalence(CPhaseGate(theta), cphase_to_rzz)

# RGate
#
#    ┌────────┐        ┌───────────────────────┐
# q: ┤ R(ϴ,φ) ├  ≡  q: ┤ U3(ϴ,φ - π/2,π/2 - φ) ├
#    └────────┘        └───────────────────────┘
q = QuantumRegister(1, "q")
theta = Parameter("theta")
phi = Parameter("phi")
def_r = QuantumCircuit(q)
def_r.append(U3Gate(theta, phi - pi / 2, -phi + pi / 2), [q[0]])
_sel.add_equivalence(RGate(theta, phi), def_r)

# IGate
q = QuantumRegister(1, "q")
def_id = QuantumCircuit(q)
def_id.append(UGate(0, 0, 0), [q[0]])
_sel.add_equivalence(IGate(), def_id)

q = QuantumRegister(1, "q")
def_id_rx = QuantumCircuit(q)
def_id_rx.append(RXGate(0), [q[0]])
_sel.add_equivalence(IGate(), def_id_rx)

q = QuantumRegister(1, "q")
def_id_ry = QuantumCircuit(q)
def_id_ry.append(RYGate(0), [q[0]])
_sel.add_equivalence(IGate(), def_id_ry)

q = QuantumRegister(1, "q")
def_id_rz = QuantumCircuit(q)
def_id_rz.append(RZGate(0), [q[0]])
_sel.add_equivalence(IGate(), def_id_rz)

# RCCXGate
#
#      ┌───────┐
# q_0: ┤0      ├     q_0: ────────────────────────■────────────────────────
#      │       │                                  │
# q_1: ┤1 Rccx ├  ≡  q_1: ────────────■───────────┼─────────■──────────────
#      │       │          ┌───┐┌───┐┌─┴─┐┌─────┐┌─┴─┐┌───┐┌─┴─┐┌─────┐┌───┐
# q_2: ┤2      ├     q_2: ┤ H ├┤ T ├┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ H ├
#      └───────┘          └───┘└───┘└───┘└─────┘└───┘└───┘└───┘└─────┘└───┘
q = QuantumRegister(3, "q")
def_rccx = QuantumCircuit(q)
for inst, qargs, cargs in [
    (HGate(), [q[2]], []),
    (TGate(), [q[2]], []),
    (CXGate(), [q[1], q[2]], []),
    (TdgGate(), [q[2]], []),
    (CXGate(), [q[0], q[2]], []),
    (TGate(), [q[2]], []),
    (CXGate(), [q[1], q[2]], []),
    (TdgGate(), [q[2]], []),
    (HGate(), [q[2]], []),
]:
    def_rccx.append(inst, qargs, cargs)
_sel.add_equivalence(RCCXGate(), def_rccx)

# RXGate
#
#    ┌───────┐        ┌────────┐
# q: ┤ Rx(ϴ) ├  ≡  q: ┤ R(ϴ,0) ├
#    └───────┘        └────────┘
q = QuantumRegister(1, "q")
theta = Parameter("theta")
def_rx = QuantumCircuit(q)
def_rx.append(RGate(theta, 0), [q[0]], [])
_sel.add_equivalence(RXGate(theta), def_rx)

# CRXGate
#
# q_0: ────■────     q_0: ─────────────■────────────────────■────────────────────
#      ┌───┴───┐  ≡       ┌─────────┐┌─┴─┐┌──────────────┐┌─┴─┐┌────────────────┐
# q_1: ┤ Rx(ϴ) ├     q_1: ┤ U1(π/2) ├┤ X ├┤ U3(-ϴ/2,0,0) ├┤ X ├┤ U3(ϴ/2,-π/2,0) ├
#      └───────┘          └─────────┘└───┘└──────────────┘└───┘└────────────────┘
q = QuantumRegister(2, "q")
theta = Parameter("theta")
def_crx = QuantumCircuit(q)
for inst, qargs, cargs in [
    (U1Gate(pi / 2), [q[1]], []),
    (CXGate(), [q[0], q[1]], []),
    (U3Gate(-theta / 2, 0, 0), [q[1]], []),
    (CXGate(), [q[0], q[1]], []),
    (U3Gate(theta / 2, -pi / 2, 0), [q[1]], []),
]:
    def_crx.append(inst, qargs, cargs)
_sel.add_equivalence(CRXGate(theta), def_crx)

# CRXGate
#
# q_0: ────■────     q_0: ───────■────────────────■────────────────────
#      ┌───┴───┐  ≡       ┌───┐┌─┴─┐┌──────────┐┌─┴─┐┌─────────┐┌─────┐
# q_1: ┤ Rx(ϴ) ├     q_1: ┤ S ├┤ X ├┤ Ry(-ϴ/2) ├┤ X ├┤ Ry(ϴ/2) ├┤ Sdg ├
#      └───────┘          └───┘└───┘└──────────┘└───┘└─────────┘└─────┘
q = QuantumRegister(2, "q")
theta = Parameter("theta")
crx_to_srycx = QuantumCircuit(q)
for inst, qargs, cargs in [
    (SGate(), [q[1]], []),
    (CXGate(), [q[0], q[1]], []),
    (RYGate(-theta / 2), [q[1]], []),
    (CXGate(), [q[0], q[1]], []),
    (RYGate(theta / 2), [q[1]], []),
    (SdgGate(), [q[1]], []),
]:
    crx_to_srycx.append(inst, qargs, cargs)
_sel.add_equivalence(CRXGate(theta), crx_to_srycx)

# CRX in terms of one RXX
#                          ┌───┐   ┌────────────┐┌───┐
# q_0: ────■────   q_0: ───┤ H ├───┤0           ├┤ H ├
#      ┌───┴───┐ ≡      ┌──┴───┴──┐│  Rxx(-ϴ/2) │└───┘
# q_1: ┤ Rx(ϴ) ├   q_1: ┤ Rx(ϴ/2) ├┤1           ├─────
#      └───────┘        └─────────┘└────────────┘
theta = Parameter("theta")
crx_to_rxx = QuantumCircuit(2)
crx_to_rxx.h(0)
crx_to_rxx.rx(theta / 2, 1)
crx_to_rxx.rxx(-theta / 2, 0, 1)
crx_to_rxx.h(0)
_sel.add_equivalence(CRXGate(theta), crx_to_rxx)

# CRX to CRZ
#
# q_0: ────■────     q_0: ─────────■─────────
#      ┌───┴───┐  ≡       ┌───┐┌───┴───┐┌───┐
# q_1: ┤ Rx(ϴ) ├     q_1: ┤ H ├┤ Rz(ϴ) ├┤ H ├
#      └───────┘          └───┘└───────┘└───┘
theta = Parameter("theta")
crx_to_crz = QuantumCircuit(2)
crx_to_crz.h(1)
crx_to_crz.crz(theta, 0, 1)
crx_to_crz.h(1)
_sel.add_equivalence(CRXGate(theta), crx_to_crz)

# RXXGate
#
#      ┌─────────┐          ┌───┐                   ┌───┐
# q_0: ┤0        ├     q_0: ┤ H ├──■─────────────■──┤ H ├
#      │  Rxx(ϴ) │  ≡       ├───┤┌─┴─┐┌───────┐┌─┴─┐├───┤
# q_1: ┤1        ├     q_1: ┤ H ├┤ X ├┤ Rz(ϴ) ├┤ X ├┤ H ├
#      └─────────┘          └───┘└───┘└───────┘└───┘└───┘
q = QuantumRegister(2, "q")
theta = Parameter("theta")
def_rxx = QuantumCircuit(q)
for inst, qargs, cargs in [
    (HGate(), [q[0]], []),
    (HGate(), [q[1]], []),
    (CXGate(), [q[0], q[1]], []),
    (RZGate(theta), [q[1]], []),
    (CXGate(), [q[0], q[1]], []),
    (HGate(), [q[1]], []),
    (HGate(), [q[0]], []),
]:
    def_rxx.append(inst, qargs, cargs)
_sel.add_equivalence(RXXGate(theta), def_rxx)

# RXX to RZX
#      ┌─────────┐        ┌───┐┌─────────┐┌───┐
# q_0: ┤0        ├   q_0: ┤ H ├┤0        ├┤ H ├
#      │  Rxx(ϴ) │ ≡      └───┘│  Rzx(ϴ) │└───┘
# q_1: ┤1        ├   q_1: ─────┤1        ├─────
#      └─────────┘             └─────────┘
theta = Parameter("theta")
rxx_to_rzx = QuantumCircuit(2)
rxx_to_rzx.h(0)
rxx_to_rzx.rzx(theta, 0, 1)
rxx_to_rzx.h(0)
_sel.add_equivalence(RXXGate(theta), rxx_to_rzx)


# RXX to RZZ
q = QuantumRegister(2, "q")
theta = Parameter("theta")
rxx_to_rzz = QuantumCircuit(q)
for inst, qargs, cargs in [
    (HGate(), [q[0]], []),
    (HGate(), [q[1]], []),
    (RZZGate(theta), [q[0], q[1]], []),
    (HGate(), [q[0]], []),
    (HGate(), [q[1]], []),
]:
    rxx_to_rzz.append(inst, qargs, cargs)
_sel.add_equivalence(RXXGate(theta), rxx_to_rzz)

# RZXGate
#
#      ┌─────────┐
# q_0: ┤0        ├     q_0: ───────■─────────────■───────
#      │  Rzx(ϴ) │  ≡       ┌───┐┌─┴─┐┌───────┐┌─┴─┐┌───┐
# q_1: ┤1        ├     q_1: ┤ H ├┤ X ├┤ Rz(ϴ) ├┤ X ├┤ H ├
#      └─────────┘          └───┘└───┘└───────┘└───┘└───┘
q = QuantumRegister(2, "q")
theta = Parameter("theta")
def_rzx = QuantumCircuit(q)
for inst, qargs, cargs in [
    (HGate(), [q[1]], []),
    (CXGate(), [q[0], q[1]], []),
    (RZGate(theta), [q[1]], []),
    (CXGate(), [q[0], q[1]], []),
    (HGate(), [q[1]], []),
]:
    def_rzx.append(inst, qargs, cargs)
_sel.add_equivalence(RZXGate(theta), def_rzx)

# RZXGate to RZZGate
#      ┌─────────┐
# q_0: ┤0        ├     q_0: ──────■───────────
#      │  Rzx(ϴ) │  ≡       ┌───┐ │ZZ(ϴ) ┌───┐
# q_1: ┤1        ├     q_1: ┤ H ├─■──────┤ H ├
#      └─────────┘          └───┘        └───┘
theta = Parameter("theta")
rzx_to_rzz = QuantumCircuit(2)
rzx_to_rzz.h(1)
rzx_to_rzz.rzz(theta, 0, 1)
rzx_to_rzz.h(1)
_sel.add_equivalence(RZXGate(theta), rzx_to_rzz)


# RYGate
#
#    ┌───────┐        ┌──────────┐
# q: ┤ Ry(ϴ) ├  ≡  q: ┤ R(ϴ,π/2) ├
#    └───────┘        └──────────┘
q = QuantumRegister(1, "q")
theta = Parameter("theta")
def_ry = QuantumCircuit(q)
def_ry.append(RGate(theta, pi / 2), [q[0]], [])
_sel.add_equivalence(RYGate(theta), def_ry)

q = QuantumRegister(1, "q")
ry_to_rx = QuantumCircuit(q)
ry_to_rx.sdg(0)
ry_to_rx.rx(theta, 0)
ry_to_rx.s(0)
_sel.add_equivalence(RYGate(theta), ry_to_rx)

# CRYGate
#
# q_0: ────■────      q_0: ─────────────■────────────────■──
#      ┌───┴───┐   ≡       ┌─────────┐┌─┴─┐┌──────────┐┌─┴─┐
# q_1: ┤ Ry(ϴ) ├      q_1: ┤ Ry(ϴ/2) ├┤ X ├┤ Ry(-ϴ/2) ├┤ X ├
#      └───────┘           └─────────┘└───┘└──────────┘└───┘
q = QuantumRegister(2, "q")
theta = Parameter("theta")
def_cry = QuantumCircuit(q)
for inst, qargs, cargs in [
    (RYGate(theta / 2), [q[1]], []),
    (CXGate(), [q[0], q[1]], []),
    (RYGate(-theta / 2), [q[1]], []),
    (CXGate(), [q[0], q[1]], []),
]:
    def_cry.append(inst, qargs, cargs)
_sel.add_equivalence(CRYGate(theta), def_cry)

# CRY to CRZ
#
# q_0: ────■────     q_0: ───────────────■────────────────
#      ┌───┴───┐  ≡       ┌─────────┐┌───┴───┐┌──────────┐
# q_1: ┤ Ry(ϴ) ├     q_1: ┤ Rx(π/2) ├┤ Rz(ϴ) ├┤ Rx(-π/2) ├
#      └───────┘          └─────────┘└───────┘└──────────┘
theta = Parameter("theta")
cry_to_crz = QuantumCircuit(2)
cry_to_crz.rx(pi / 2, 1)
cry_to_crz.crz(theta, 0, 1)
cry_to_crz.rx(-pi / 2, 1)
_sel.add_equivalence(CRYGate(theta), cry_to_crz)

# CRY to CRZ
#
# q_0: ────■────     q_0: ────────────────────■─────────────────────
#      ┌───┴───┐  ≡       ┌───┐┌─────────┐┌───┴───┐┌──────────┐┌───┐
# q_1: ┤ Ry(ϴ) ├     q_1: ┤ H ├┤ Rz(π/2) ├┤ Rx(ϴ) ├┤ Rz(-π/2) ├┤ H ├
#      └───────┘          └───┘└─────────┘└───────┘└──────────┘└───┘
theta = Parameter("theta")
cry_to_crx = QuantumCircuit(2)
cry_to_crx.h(1)
cry_to_crx.rz(pi / 2, 1)
cry_to_crx.crx(theta, 0, 1)
cry_to_crx.rz(-pi / 2, 1)
cry_to_crx.h(1)
_sel.add_equivalence(CRYGate(theta), cry_to_crx)

# CRY to RZZ
#
# q_0: ────■────    q_0: ────────────────────────■───────────────────
#      ┌───┴───┐  ≡      ┌─────┐┌─────────┐┌───┐ │ZZ(-ϴ/2) ┌───┐┌───┐
# q_1: ┤ Ry(ϴ) ├    q_1: ┤ Sdg ├┤ Rx(ϴ/2) ├┤ H ├─■─────────┤ H ├┤ S ├
#      └───────┘         └─────┘└─────────┘└───┘           └───┘└───┘
cry_to_rzz = QuantumCircuit(2)
cry_to_rzz.sdg(1)
cry_to_rzz.rx(theta / 2, 1)
cry_to_rzz.h(1)
cry_to_rzz.rzz(-theta / 2, 0, 1)
cry_to_rzz.h(1)
cry_to_rzz.s(1)
_sel.add_equivalence(CRYGate(theta), cry_to_rzz)

# RYYGate
#
#      ┌─────────┐          ┌─────────┐                   ┌──────────┐
# q_0: ┤0        ├     q_0: ┤ Rx(π/2) ├──■─────────────■──┤ Rx(-π/2) ├
#      │  Ryy(ϴ) │  ≡       ├─────────┤┌─┴─┐┌───────┐┌─┴─┐├──────────┤
# q_1: ┤1        ├     q_1: ┤ Rx(π/2) ├┤ X ├┤ Rz(ϴ) ├┤ X ├┤ Rx(-π/2) ├
#      └─────────┘          └─────────┘└───┘└───────┘└───┘└──────────┘
q = QuantumRegister(2, "q")
theta = Parameter("theta")
def_ryy = QuantumCircuit(q)
for inst, qargs, cargs in [
    (RXGate(pi / 2), [q[0]], []),
    (RXGate(pi / 2), [q[1]], []),
    (CXGate(), [q[0], q[1]], []),
    (RZGate(theta), [q[1]], []),
    (CXGate(), [q[0], q[1]], []),
    (RXGate(-pi / 2), [q[0]], []),
    (RXGate(-pi / 2), [q[1]], []),
]:
    def_ryy.append(inst, qargs, cargs)
_sel.add_equivalence(RYYGate(theta), def_ryy)

# RYY to RZZ
q = QuantumRegister(2, "q")
theta = Parameter("theta")
ryy_to_rzz = QuantumCircuit(q)
for inst, qargs, cargs in [
    (RXGate(pi / 2), [q[0]], []),
    (RXGate(pi / 2), [q[1]], []),
    (RZZGate(theta), [q[0], q[1]], []),
    (RXGate(-pi / 2), [q[0]], []),
    (RXGate(-pi / 2), [q[1]], []),
]:
    ryy_to_rzz.append(inst, qargs, cargs)
_sel.add_equivalence(RYYGate(theta), ryy_to_rzz)

# RYY to RXX
q = QuantumRegister(2, "q")
theta = Parameter("theta")
ryy_to_rxx = QuantumCircuit(q)
for inst, qargs, cargs in [
    (SdgGate(), [q[0]], []),
    (SdgGate(), [q[1]], []),
    (RXXGate(theta), [q[0], q[1]], []),
    (SGate(), [q[0]], []),
    (SGate(), [q[1]], []),
]:
    ryy_to_rxx.append(inst, qargs, cargs)
_sel.add_equivalence(RYYGate(theta), ryy_to_rxx)

# RZGate
#                  global phase: -ϴ/2
#    ┌───────┐        ┌───────┐
# q: ┤ Rz(ϴ) ├  ≡  q: ┤ U1(ϴ) ├
#    └───────┘        └───────┘
q = QuantumRegister(1, "q")
theta = Parameter("theta")
def_rz = QuantumCircuit(q, global_phase=-theta / 2)
def_rz.append(U1Gate(theta), [q[0]], [])
_sel.add_equivalence(RZGate(theta), def_rz)

# RZGate
#
#    ┌───────┐        ┌────┐┌────────┐┌──────┐
# q: ┤ Rz(ϴ) ├  ≡  q: ┤ √X ├┤ Ry(-ϴ) ├┤ √Xdg ├
#    └───────┘        └────┘└────────┘└──────┘
q = QuantumRegister(1, "q")
rz_to_sxry = QuantumCircuit(q)
rz_to_sxry.sx(0)
rz_to_sxry.ry(-theta, 0)
rz_to_sxry.sxdg(0)
_sel.add_equivalence(RZGate(theta), rz_to_sxry)

q = QuantumRegister(1, "q")
rz_to_rx = QuantumCircuit(q)
rz_to_rx.h(0)
rz_to_rx.rx(theta, 0)
rz_to_rx.h(0)
_sel.add_equivalence(RZGate(theta), rz_to_rx)

# CRZGate
#
# q_0: ────■────     q_0: ─────────────■────────────────■──
#      ┌───┴───┐  ≡       ┌─────────┐┌─┴─┐┌──────────┐┌─┴─┐
# q_1: ┤ Rz(ϴ) ├     q_1: ┤ Rz(ϴ/2) ├┤ X ├┤ Rz(-ϴ/2) ├┤ X ├
#      └───────┘          └─────────┘└───┘└──────────┘└───┘
q = QuantumRegister(2, "q")
theta = Parameter("theta")
def_crz = QuantumCircuit(q)
for inst, qargs, cargs in [
    (RZGate(theta / 2), [q[1]], []),
    (CXGate(), [q[0], q[1]], []),
    (RZGate(-theta / 2), [q[1]], []),
    (CXGate(), [q[0], q[1]], []),
]:
    def_crz.append(inst, qargs, cargs)
_sel.add_equivalence(CRZGate(theta), def_crz)

# CRZ to CRY
#
# q_0: ────■────     q_0: ────────────────■───────────────
#      ┌───┴───┐  ≡       ┌──────────┐┌───┴───┐┌─────────┐
# q_1: ┤ Rz(ϴ) ├     q_1: ┤ Rx(-π/2) ├┤ Ry(ϴ) ├┤ Rx(π/2) ├
#      └───────┘          └──────────┘└───────┘└─────────┘
theta = Parameter("theta")
crz_to_cry = QuantumCircuit(2)
crz_to_cry.rx(-pi / 2, 1)
crz_to_cry.cry(theta, 0, 1)
crz_to_cry.rx(pi / 2, 1)
_sel.add_equivalence(CRZGate(theta), crz_to_cry)

# CRZ to CRX
#
# q_0: ────■────     q_0: ─────────■─────────
#      ┌───┴───┐  ≡       ┌───┐┌───┴───┐┌───┐
# q_1: ┤ Rz(ϴ) ├     q_1: ┤ H ├┤ Rx(ϴ) ├┤ H ├
#      └───────┘          └───┘└───────┘└───┘
theta = Parameter("theta")
crz_to_crx = QuantumCircuit(2)
crz_to_crx.h(1)
crz_to_crx.crx(theta, 0, 1)
crz_to_crx.h(1)
_sel.add_equivalence(CRZGate(theta), crz_to_crx)

# CRZ to RZZ
#
# q_0: ────■────    q_0: ────────────■────────
#      ┌───┴───┐  ≡      ┌─────────┐ │ZZ(-ϴ/2)
# q_1: ┤ Rz(ϴ) ├    q_1: ┤ Rz(ϴ/2) ├─■────────
#      └───────┘         └─────────┘
theta = Parameter("theta")
crz_to_rzz = QuantumCircuit(2)
crz_to_rzz.rz(theta / 2, 1)
crz_to_rzz.rzz(-theta / 2, 0, 1)
_sel.add_equivalence(CRZGate(theta), crz_to_rzz)

# RZZGate
#
# q_0: ─■─────     q_0: ──■─────────────■──
#       │ZZ(ϴ)  ≡       ┌─┴─┐┌───────┐┌─┴─┐
# q_1: ─■─────     q_1: ┤ X ├┤ Rz(ϴ) ├┤ X ├
#                       └───┘└───────┘└───┘
q = QuantumRegister(2, "q")
theta = Parameter("theta")
def_rzz = QuantumCircuit(q)
for inst, qargs, cargs in [
    (CXGate(), [q[0], q[1]], []),
    (RZGate(theta), [q[1]], []),
    (CXGate(), [q[0], q[1]], []),
]:
    def_rzz.append(inst, qargs, cargs)
_sel.add_equivalence(RZZGate(theta), def_rzz)

# RZZ to RXX
q = QuantumRegister(2, "q")
theta = Parameter("theta")
rzz_to_rxx = QuantumCircuit(q)
for inst, qargs, cargs in [
    (HGate(), [q[0]], []),
    (HGate(), [q[1]], []),
    (RXXGate(theta), [q[0], q[1]], []),
    (HGate(), [q[0]], []),
    (HGate(), [q[1]], []),
]:
    rzz_to_rxx.append(inst, qargs, cargs)
_sel.add_equivalence(RZZGate(theta), rzz_to_rxx)

# RZZ to RZX
#                          ┌─────────┐
# q_0: ─■─────   q_0: ─────┤0        ├─────
#       │ZZ(ϴ) ≡      ┌───┐│  Rzx(ϴ) │┌───┐
# q_1: ─■─────   q_1: ┤ H ├┤1        ├┤ H ├
#                     └───┘└─────────┘└───┘
theta = Parameter("theta")
rzz_to_rzx = QuantumCircuit(2)
rzz_to_rzx.h(1)
rzz_to_rzx.rzx(theta, 0, 1)
rzz_to_rzx.h(1)
_sel.add_equivalence(RZZGate(theta), rzz_to_rzx)

# RZZ to CPhase
#
#                 global phase: ϴ/2
#                                ┌───────┐
#  q_0: ─■─────   q_0: ─■────────┤ Rz(ϴ) ├
#        │ZZ(ϴ) ≡       │P(-2*ϴ) ├───────┤
#  q_1: ─■─────   q_1: ─■────────┤ Rz(ϴ) ├
#                                └───────┘
theta = Parameter("theta")
rzz_to_cphase = QuantumCircuit(2, global_phase=theta / 2)
rzz_to_cphase.cp(-theta * 2, 0, 1)
rzz_to_cphase.rz(theta, 0)
rzz_to_cphase.rz(theta, 1)
_sel.add_equivalence(RZZGate(theta), rzz_to_cphase)

# RZZ to RYY
q = QuantumRegister(2, "q")
theta = Parameter("theta")
rzz_to_ryy = QuantumCircuit(q)
for inst, qargs, cargs in [
    (RXGate(-pi / 2), [q[0]], []),
    (RXGate(-pi / 2), [q[1]], []),
    (RYYGate(theta), [q[0], q[1]], []),
    (RXGate(pi / 2), [q[0]], []),
    (RXGate(pi / 2), [q[1]], []),
]:
    rzz_to_ryy.append(inst, qargs, cargs)
_sel.add_equivalence(RZZGate(theta), rzz_to_ryy)

# RZXGate
#
#      ┌─────────┐
# q_0: ┤0        ├     q_0: ───────■─────────────■───────
#      │  Rzx(ϴ) │  ≡       ┌───┐┌─┴─┐┌───────┐┌─┴─┐┌───┐
# q_1: ┤1        ├     q_1: ┤ H ├┤ X ├┤ Rz(ϴ) ├┤ X ├┤ H ├
#      └─────────┘          └───┘└───┘└───────┘└───┘└───┘
q = QuantumRegister(2, "q")
theta = Parameter("theta")
def_rzx = QuantumCircuit(q)
for inst, qargs, cargs in [
    (HGate(), [q[1]], []),
    (CXGate(), [q[0], q[1]], []),
    (RZGate(theta), [q[1]], []),
    (CXGate(), [q[0], q[1]], []),
    (HGate(), [q[1]], []),
]:
    def_rzx.append(inst, qargs, cargs)
_sel.add_equivalence(RZXGate(theta), def_rzx)

# ECRGate
#
#      ┌──────┐          ┌───────────┐┌───┐┌────────────┐
# q_0: ┤0     ├     q_0: ┤0          ├┤ X ├┤0           ├
#      │  Ecr │  ≡       │  Rzx(π/4) │└───┘│  Rzx(-π/4) │
# q_1: ┤1     ├     q_1: ┤1          ├─────┤1           ├
#      └──────┘          └───────────┘     └────────────┘
q = QuantumRegister(2, "q")
def_ecr = QuantumCircuit(q)
for inst, qargs, cargs in [
    (RZXGate(pi / 4), [q[0], q[1]], []),
    (XGate(), [q[0]], []),
    (RZXGate(-pi / 4), [q[0], q[1]], []),
]:
    def_ecr.append(inst, qargs, cargs)
_sel.add_equivalence(ECRGate(), def_ecr)

# ECRGate decomposed to Clifford gates (up to a global phase)
#
#                  global phase: 7π/4
#      ┌──────┐         ┌───┐      ┌───┐
# q_0: ┤0     ├    q_0: ┤ S ├───■──┤ X ├
#      │  Ecr │  ≡      ├───┴┐┌─┴─┐└───┘
# q_1: ┤1     ├    q_1: ┤ √X ├┤ X ├─────
#      └──────┘         └────┘└───┘

q = QuantumRegister(2, "q")
def_ecr_cliff = QuantumCircuit(q, global_phase=-pi / 4)
for inst, qargs, cargs in [
    (SGate(), [q[0]], []),
    (SXGate(), [q[1]], []),
    (CXGate(), [q[0], q[1]], []),
    (XGate(), [q[0]], []),
]:
    def_ecr_cliff.append(inst, qargs, cargs)
_sel.add_equivalence(ECRGate(), def_ecr_cliff)

# CXGate decomposed using an ECRGate and Clifford 1-qubit gates
#                global phase: π/4
# q_0: ──■──          ┌─────┐ ┌──────┐┌───┐
#      ┌─┴─┐  ≡  q_0: ┤ Sdg ├─┤0     ├┤ X ├
# q_1: ┤ X ├          ├─────┴┐│  Ecr │└───┘
#      └───┘     q_1: ┤ √Xdg ├┤1     ├─────
#                     └──────┘└──────┘

q = QuantumRegister(2, "q")
def_ecr_to_cx_cliff = QuantumCircuit(q, global_phase=pi / 4)
for inst, qargs, cargs in [
    (SdgGate(), [q[0]], []),
    (SXdgGate(), [q[1]], []),
    (ECRGate(), [q[0], q[1]], []),
    (XGate(), [q[0]], []),
]:
    def_ecr_to_cx_cliff.append(inst, qargs, cargs)
_sel.add_equivalence(CXGate(), def_ecr_to_cx_cliff)

# SGate
#
#    ┌───┐        ┌─────────┐
# q: ┤ S ├  ≡  q: ┤ U1(π/2) ├
#    └───┘        └─────────┘
q = QuantumRegister(1, "q")
def_s = QuantumCircuit(q)
def_s.append(U1Gate(pi / 2), [q[0]], [])
_sel.add_equivalence(SGate(), def_s)

# SdgGate
#
#    ┌─────┐        ┌──────────┐
# q: ┤ Sdg ├  ≡  q: ┤ U1(-π/2) ├
#    └─────┘        └──────────┘
q = QuantumRegister(1, "q")
def_sdg = QuantumCircuit(q)
def_sdg.append(U1Gate(-pi / 2), [q[0]], [])
_sel.add_equivalence(SdgGate(), def_sdg)

# CSGate
#
# q_0: ──■──   q_0: ───────■────────
#      ┌─┴─┐        ┌───┐┌─┴──┐┌───┐
# q_1: ┤ S ├ = q_1: ┤ H ├┤ Sx ├┤ H ├
#      └───┘        └───┘└────┘└───┘
q = QuantumRegister(2, "q")
def_cs = QuantumCircuit(q)
def_cs.append(HGate(), [q[1]], [])
def_cs.append(CSXGate(), [q[0], q[1]], [])
def_cs.append(HGate(), [q[1]], [])
_sel.add_equivalence(CSGate(), def_cs)

# CSdgGate
#
# q_0: ───■───   q_0: ───────■────■────────
#      ┌──┴──┐        ┌───┐┌─┴─┐┌─┴──┐┌───┐
# q_1: ┤ Sdg ├ = q_1: ┤ H ├┤ X ├┤ Sx ├┤ H ├
#      └─────┘        └───┘└───┘└────┘└───┘
q = QuantumRegister(2, "q")
def_csdg = QuantumCircuit(q)
def_csdg.append(HGate(), [q[1]], [])
def_csdg.append(CXGate(), [q[0], q[1]], [])
def_csdg.append(CSXGate(), [q[0], q[1]], [])
def_csdg.append(HGate(), [q[1]], [])
_sel.add_equivalence(CSdgGate(), def_csdg)

# SdgGate
#
#    ┌─────┐        ┌───┐┌───┐
# q: ┤ Sdg ├  ≡  q: ┤ S ├┤ Z ├
#    └─────┘        └───┘└───┘
q = QuantumRegister(1, "q")
def_sdg = QuantumCircuit(q)
for inst, qargs, cargs in [
    (SGate(), [q[0]], []),
    (ZGate(), [q[0]], []),
]:
    def_sdg.append(inst, qargs, cargs)
_sel.add_equivalence(SdgGate(), def_sdg)

# SdgGate
#
#    ┌─────┐        ┌───┐┌───┐
# q: ┤ Sdg ├  ≡  q: ┤ Z ├┤ S ├
#    └─────┘        └───┘└───┘
q = QuantumRegister(1, "q")
def_sdg = QuantumCircuit(q)
for inst, qargs, cargs in [
    (ZGate(), [q[0]], []),
    (SGate(), [q[0]], []),
]:
    def_sdg.append(inst, qargs, cargs)
_sel.add_equivalence(SdgGate(), def_sdg)

# SdgGate
#
#    ┌─────┐        ┌───┐┌───┐┌───┐
# q: ┤ Sdg ├  ≡  q: ┤ S ├┤ S ├┤ S ├
#    └─────┘        └───┘└───┘└───┘
q = QuantumRegister(1, "q")
def_sdg = QuantumCircuit(q)
for inst, qargs, cargs in [
    (SGate(), [q[0]], []),
    (SGate(), [q[0]], []),
    (SGate(), [q[0]], []),
]:
    def_sdg.append(inst, qargs, cargs)
_sel.add_equivalence(SdgGate(), def_sdg)

# SwapGate
#                        ┌───┐
# q_0: ─X─     q_0: ──■──┤ X ├──■──
#       │   ≡       ┌─┴─┐└─┬─┘┌─┴─┐
# q_1: ─X─     q_1: ┤ X ├──■──┤ X ├
#                   └───┘     └───┘
q = QuantumRegister(2, "q")
def_swap = QuantumCircuit(q)
for inst, qargs, cargs in [
    (CXGate(), [q[0], q[1]], []),
    (CXGate(), [q[1], q[0]], []),
    (CXGate(), [q[0], q[1]], []),
]:
    def_swap.append(inst, qargs, cargs)
_sel.add_equivalence(SwapGate(), def_swap)

# SwapGate
#
# q_0: ─X─
#       │   ≡
# q_1: ─X─
#
#      ┌──────────┐┌──────┐   ┌────┐   ┌──────┐┌──────────┐┌──────┐
# q_0: ┤ Rz(-π/2) ├┤0     ├───┤ √X ├───┤1     ├┤ Rz(-π/2) ├┤0     ├
#      └──┬────┬──┘│  Ecr │┌──┴────┴──┐│  Ecr │└──┬────┬──┘│  Ecr │
# q_1: ───┤ √X ├───┤1     ├┤ Rz(-π/2) ├┤0     ├───┤ √X ├───┤1     ├
#         └────┘   └──────┘└──────────┘└──────┘   └────┘   └──────┘
#
q = QuantumRegister(2, "q")
def_swap_ecr = QuantumCircuit(q)
def_swap_ecr.rz(-pi / 2, 0)
def_swap_ecr.sx(1)
def_swap_ecr.ecr(0, 1)
def_swap_ecr.rz(-pi / 2, 1)
def_swap_ecr.sx(0)
def_swap_ecr.ecr(1, 0)
def_swap_ecr.rz(-pi / 2, 0)
def_swap_ecr.sx(1)
def_swap_ecr.ecr(0, 1)
_sel.add_equivalence(SwapGate(), def_swap_ecr)

# SwapGate
#
# q_0: ─X─
#       │   ≡
# q_1: ─X─
#
# global phase: 3π/2
#      ┌────┐   ┌────┐   ┌────┐
# q_0: ┤ √X ├─■─┤ √X ├─■─┤ √X ├─■─
#      ├────┤ │ ├────┤ │ ├────┤ │
# q_1: ┤ √X ├─■─┤ √X ├─■─┤ √X ├─■─
#      └────┘   └────┘   └────┘
q = QuantumRegister(2, "q")
def_swap_cz = QuantumCircuit(q, global_phase=-pi / 2)
def_swap_cz.sx(0)
def_swap_cz.sx(1)
def_swap_cz.cz(0, 1)
def_swap_cz.sx(0)
def_swap_cz.sx(1)
def_swap_cz.cz(0, 1)
def_swap_cz.sx(0)
def_swap_cz.sx(1)
def_swap_cz.cz(0, 1)
_sel.add_equivalence(SwapGate(), def_swap_cz)

# iSwapGate
#
#      ┌────────┐          ┌───┐┌───┐     ┌───┐
# q_0: ┤0       ├     q_0: ┤ S ├┤ H ├──■──┤ X ├─────
#      │  Iswap │  ≡       ├───┤└───┘┌─┴─┐└─┬─┘┌───┐
# q_1: ┤1       ├     q_1: ┤ S ├─────┤ X ├──■──┤ H ├
#      └────────┘          └───┘     └───┘     └───┘
q = QuantumRegister(2, "q")
def_iswap = QuantumCircuit(q)
for inst, qargs, cargs in [
    (SGate(), [q[0]], []),
    (SGate(), [q[1]], []),
    (HGate(), [q[0]], []),
    (CXGate(), [q[0], q[1]], []),
    (CXGate(), [q[1], q[0]], []),
    (HGate(), [q[1]], []),
]:
    def_iswap.append(inst, qargs, cargs)
_sel.add_equivalence(iSwapGate(), def_iswap)

# SXGate
#               global phase: π/4
#    ┌────┐        ┌─────┐┌───┐┌─────┐
# q: ┤ √X ├  ≡  q: ┤ Sdg ├┤ H ├┤ Sdg ├
#    └────┘        └─────┘└───┘└─────┘
q = QuantumRegister(1, "q")
def_sx = QuantumCircuit(q, global_phase=pi / 4)
for inst, qargs, cargs in [(SdgGate(), [q[0]], []), (HGate(), [q[0]], []), (SdgGate(), [q[0]], [])]:
    def_sx.append(inst, qargs, cargs)
_sel.add_equivalence(SXGate(), def_sx)

# HGate decomposed into SXGate and SGate
#              global phase: -π/4
#    ┌───┐        ┌───┐┌────┐┌───┐
# q: ┤ H ├  ≡  q: ┤ S ├┤ √X ├┤ S ├
#    └───┘        └───┘└────┘└───┘
q = QuantumRegister(1, "q")
def_h_to_sx = QuantumCircuit(q, global_phase=-pi / 4)
for inst, qargs, cargs in [(SGate(), [q[0]], []), (SXGate(), [q[0]], []), (SGate(), [q[0]], [])]:
    def_h_to_sx.append(inst, qargs, cargs)
_sel.add_equivalence(HGate(), def_h_to_sx)

# SXGate
#               global phase: π/4
#    ┌────┐        ┌─────────┐
# q: ┤ √X ├  ≡  q: ┤ Rx(π/2) ├
#    └────┘        └─────────┘
q = QuantumRegister(1, "q")
sx_to_rx = QuantumCircuit(q, global_phase=pi / 4)
sx_to_rx.rx(pi / 2, 0)
_sel.add_equivalence(SXGate(), sx_to_rx)

# SXdgGate
#                 global phase: 7π/4
#    ┌──────┐        ┌───┐┌───┐┌───┐
# q: ┤ √Xdg ├  ≡  q: ┤ S ├┤ H ├┤ S ├
#    └──────┘        └───┘└───┘└───┘
q = QuantumRegister(1, "q")
def_sxdg = QuantumCircuit(q, global_phase=-pi / 4)
for inst, qargs, cargs in [(SGate(), [q[0]], []), (HGate(), [q[0]], []), (SGate(), [q[0]], [])]:
    def_sxdg.append(inst, qargs, cargs)
_sel.add_equivalence(SXdgGate(), def_sxdg)

# HGate decomposed into SXdgGate and SdgGate
#              global phase: π/4
#    ┌───┐        ┌─────┐┌──────┐┌─────┐
# q: ┤ H ├  ≡  q: ┤ Sdg ├┤ √Xdg ├┤ Sdg ├
#    └───┘        └─────┘└──────┘└─────┘
q = QuantumRegister(1, "q")
def_h_to_sxdg = QuantumCircuit(q, global_phase=pi / 4)
for inst, qargs, cargs in [
    (SdgGate(), [q[0]], []),
    (SXdgGate(), [q[0]], []),
    (SdgGate(), [q[0]], []),
]:
    def_h_to_sxdg.append(inst, qargs, cargs)
_sel.add_equivalence(HGate(), def_h_to_sxdg)

# SXdgGate
#                 global phase: 7π/4
#    ┌──────┐        ┌──────────┐
# q: ┤ √Xdg ├  ≡  q: ┤ Rx(-π/2) ├
#    └──────┘        └──────────┘
q = QuantumRegister(1, "q")
sxdg_to_rx = QuantumCircuit(q, global_phase=-pi / 4)
sxdg_to_rx.rx(-pi / 2, 0)
_sel.add_equivalence(SXdgGate(), sxdg_to_rx)

# CSXGate
#
# q_0: ──■───     q_0: ──────■─────────────
#      ┌─┴──┐  ≡       ┌───┐ │U1(π/2) ┌───┐
# q_1: ┤ Sx ├     q_1: ┤ H ├─■────────┤ H ├
#      └────┘          └───┘          └───┘
q = QuantumRegister(2, "q")
def_csx = QuantumCircuit(q)
for inst, qargs, cargs in [
    (HGate(), [q[1]], []),
    (CU1Gate(pi / 2), [q[0], q[1]], []),
    (HGate(), [q[1]], []),
]:
    def_csx.append(inst, qargs, cargs)
_sel.add_equivalence(CSXGate(), def_csx)

# CSXGate
#                 global phase: π/8
#                      ┌───┐┌───────────┐ ┌─────┐  ┌───┐
# q_0: ──■───     q_0: ┤ X ├┤0          ├─┤ Tdg ├──┤ X ├
#      ┌─┴──┐  ≡       └───┘│  Rzx(π/4) │┌┴─────┴─┐└───┘
# q_1: ┤ Sx ├     q_1: ─────┤1          ├┤ sx^0.5 ├─────
#      └────┘               └───────────┘└────────┘
q = QuantumRegister(2, "q")
csx_to_zx45 = QuantumCircuit(q, global_phase=pi / 4)
for inst, qargs, cargs in [
    (XGate(), [q[0]], []),
    (RZXGate(pi / 4), [q[0], q[1]], []),
    (TdgGate(), [q[0]], []),
    (XGate(), [q[0]], []),
    (RXGate(pi / 4), [q[1]], []),
]:
    csx_to_zx45.append(inst, qargs, cargs)
_sel.add_equivalence(CSXGate(), csx_to_zx45)


# DCXGate
#
#      ┌──────┐               ┌───┐
# q_0: ┤0     ├     q_0: ──■──┤ X ├
#      │  Dcx │  ≡       ┌─┴─┐└─┬─┘
# q_1: ┤1     ├     q_1: ┤ X ├──■──
#      └──────┘          └───┘
q = QuantumRegister(2, "q")
def_dcx = QuantumCircuit(q)
for inst, qargs, cargs in [(CXGate(), [q[0], q[1]], []), (CXGate(), [q[1], q[0]], [])]:
    def_dcx.append(inst, qargs, cargs)
_sel.add_equivalence(DCXGate(), def_dcx)

# DCXGate
#
#      ┌──────┐           ┌───┐ ┌─────┐┌────────┐
# q_0: ┤0     ├     q_0: ─┤ H ├─┤ Sdg ├┤0       ├─────
#      │  Dcx │  ≡       ┌┴───┴┐└─────┘│  Iswap │┌───┐
# q_1: ┤1     ├     q_1: ┤ Sdg ├───────┤1       ├┤ H ├
#      └──────┘          └─────┘       └────────┘└───┘
q = QuantumRegister(2, "q")
dcx_to_iswap = QuantumCircuit(q)
for inst, qargs, cargs in [
    (HGate(), [q[0]], []),
    (SdgGate(), [q[0]], []),
    (SdgGate(), [q[1]], []),
    (iSwapGate(), [q[0], q[1]], []),
    (HGate(), [q[1]], []),
]:
    dcx_to_iswap.append(inst, qargs, cargs)
_sel.add_equivalence(DCXGate(), dcx_to_iswap)

# CSwapGate
#
# q_0: ─■─     q_0: ───────■───────
#       │           ┌───┐  │  ┌───┐
# q_1: ─X─  ≡  q_1: ┤ X ├──■──┤ X ├
#       │           └─┬─┘┌─┴─┐└─┬─┘
# q_2: ─X─     q_2: ──■──┤ X ├──■──
#                        └───┘
q = QuantumRegister(3, "q")
def_cswap = QuantumCircuit(q)
for inst, qargs, cargs in [
    (CXGate(), [q[2], q[1]], []),
    (CCXGate(), [q[0], q[1], q[2]], []),
    (CXGate(), [q[2], q[1]], []),
]:
    def_cswap.append(inst, qargs, cargs)
_sel.add_equivalence(CSwapGate(), def_cswap)

# TGate
#
#    ┌───┐        ┌─────────┐
# q: ┤ T ├  ≡  q: ┤ U1(π/4) ├
#    └───┘        └─────────┘
q = QuantumRegister(1, "q")
def_t = QuantumCircuit(q)
def_t.append(U1Gate(pi / 4), [q[0]], [])
_sel.add_equivalence(TGate(), def_t)

# TdgGate
#
#    ┌─────┐        ┌──────────┐
# q: ┤ Tdg ├  ≡  q: ┤ U1(-π/4) ├
#    └─────┘        └──────────┘
q = QuantumRegister(1, "q")
def_tdg = QuantumCircuit(q)
def_tdg.append(U1Gate(-pi / 4), [q[0]], [])
_sel.add_equivalence(TdgGate(), def_tdg)

# UGate
#
#    ┌──────────┐        ┌───────────┐
# q: ┤ U(θ,ϕ,λ) ├  ≡  q: ┤ U3(θ,ϕ,λ) ├
#    └──────────┘        └───────────┘
q = QuantumRegister(1, "q")
theta = Parameter("theta")
phi = Parameter("phi")
lam = Parameter("lam")
u_to_u3 = QuantumCircuit(q)
u_to_u3.append(U3Gate(theta, phi, lam), [0])
_sel.add_equivalence(UGate(theta, phi, lam), u_to_u3)

# CUGate
#                                  ┌──────┐    ┌──────────────┐     »
# q_0: ──────■───────     q_0: ────┤ P(γ) ├────┤ P(λ/2 + ϕ/2) ├──■──»
#      ┌─────┴──────┐  ≡       ┌───┴──────┴───┐└──────────────┘┌─┴─┐»
# q_1: ┤ U(θ,ϕ,λ,γ) ├     q_1: ┤ P(λ/2 - ϕ/2) ├────────────────┤ X ├»
#      └────────────┘          └──────────────┘                └───┘»
# «
# «q_0: ──────────────────────────■────────────────
# «     ┌──────────────────────┐┌─┴─┐┌────────────┐
# «q_1: ┤ U(-θ/2,ϕ,-λ/2 - ϕ/2) ├┤ X ├┤ U(θ/2,ϕ,0) ├
# «     └──────────────────────┘└───┘└────────────┘
q = QuantumRegister(2, "q")
theta = Parameter("theta")
phi = Parameter("phi")
lam = Parameter("lam")
gamma = Parameter("gamma")
def_cu = QuantumCircuit(q)
def_cu.p(gamma, 0)
def_cu.p((lam + phi) / 2, 0)
def_cu.p((lam - phi) / 2, 1)
def_cu.cx(0, 1)
def_cu.u(-theta / 2, 0, -(phi + lam) / 2, 1)
def_cu.cx(0, 1)
def_cu.u(theta / 2, phi, 0, 1)
_sel.add_equivalence(CUGate(theta, phi, lam, gamma), def_cu)

# CUGate
#                              ┌──────┐
# q_0: ──────■───────     q_0: ┤ P(γ) ├──────■──────
#      ┌─────┴──────┐  ≡       └──────┘┌─────┴─────┐
# q_1: ┤ U(θ,ϕ,λ,γ) ├     q_1: ────────┤ U3(θ,ϕ,λ) ├
#      └────────────┘                  └───────────┘
q = QuantumRegister(2, "q")
theta = Parameter("theta")
phi = Parameter("phi")
lam = Parameter("lam")
gamma = Parameter("gamma")
cu_to_cu3 = QuantumCircuit(q)
cu_to_cu3.p(gamma, 0)
cu_to_cu3.append(CU3Gate(theta, phi, lam), [0, 1])
_sel.add_equivalence(CUGate(theta, phi, lam, gamma), cu_to_cu3)

# U1Gate
#
#    ┌───────┐        ┌───────────┐
# q: ┤ U1(θ) ├  ≡  q: ┤ U3(0,0,θ) ├
#    └───────┘        └───────────┘
q = QuantumRegister(1, "q")
theta = Parameter("theta")
def_u1 = QuantumCircuit(q)
def_u1.append(U3Gate(0, 0, theta), [q[0]], [])
_sel.add_equivalence(U1Gate(theta), def_u1)

# U1Gate
#
#    ┌───────┐        ┌──────┐
# q: ┤ U1(θ) ├  ≡  q: ┤ P(0) ├
#    └───────┘        └──────┘
q = QuantumRegister(1, "q")
theta = Parameter("theta")
u1_to_phase = QuantumCircuit(q)
u1_to_phase.p(theta, 0)
_sel.add_equivalence(U1Gate(theta), u1_to_phase)

# U1Gate
#                  global phase: θ/2
#    ┌───────┐        ┌───────┐
# q: ┤ U1(θ) ├  ≡  q: ┤ Rz(θ) ├
#    └───────┘        └───────┘
q = QuantumRegister(1, "q")
theta = Parameter("theta")
u1_to_rz = QuantumCircuit(q, global_phase=theta / 2)
u1_to_rz.append(RZGate(theta), [q[0]], [])
_sel.add_equivalence(U1Gate(theta), u1_to_rz)

# CU1Gate
#                       ┌─────────┐
# q_0: ─■─────     q_0: ┤ U1(θ/2) ├──■────────────────■─────────────
#       │U1(θ)  ≡       └─────────┘┌─┴─┐┌──────────┐┌─┴─┐┌─────────┐
# q_1: ─■─────     q_1: ───────────┤ X ├┤ U1(-θ/2) ├┤ X ├┤ U1(θ/2) ├
#                                  └───┘└──────────┘└───┘└─────────┘
q = QuantumRegister(2, "q")
theta = Parameter("theta")
def_cu1 = QuantumCircuit(q)
for inst, qargs, cargs in [
    (U1Gate(theta / 2), [q[0]], []),
    (CXGate(), [q[0], q[1]], []),
    (U1Gate(-theta / 2), [q[1]], []),
    (CXGate(), [q[0], q[1]], []),
    (U1Gate(theta / 2), [q[1]], []),
]:
    def_cu1.append(inst, qargs, cargs)
_sel.add_equivalence(CU1Gate(theta), def_cu1)

# U2Gate
#
#    ┌─────────┐        ┌─────────────┐
# q: ┤ U2(ϕ,λ) ├  ≡  q: ┤ U3(π/2,ϕ,λ) ├
#    └─────────┘        └─────────────┘
q = QuantumRegister(1, "q")
phi = Parameter("phi")
lam = Parameter("lam")
def_u2 = QuantumCircuit(q)
def_u2.append(U3Gate(pi / 2, phi, lam), [q[0]], [])
_sel.add_equivalence(U2Gate(phi, lam), def_u2)

# U2Gate
#                    global phase: 7π/4
#    ┌─────────┐        ┌─────────────┐┌────┐┌─────────────┐
# q: ┤ U2(ϕ,λ) ├  ≡  q: ┤ U1(λ - π/2) ├┤ √X ├┤ U1(ϕ + π/2) ├
#    └─────────┘        └─────────────┘└────┘└─────────────┘
q = QuantumRegister(1, "q")
phi = Parameter("phi")
lam = Parameter("lam")
u2_to_u1sx = QuantumCircuit(q, global_phase=-pi / 4)
u2_to_u1sx.append(U1Gate(lam - pi / 2), [0])
u2_to_u1sx.sx(0)
u2_to_u1sx.append(U1Gate(phi + pi / 2), [0])
_sel.add_equivalence(U2Gate(phi, lam), u2_to_u1sx)

# U3Gate
#                         global phase: λ/2 + ϕ/2 - π/2
#    ┌───────────┐        ┌───────┐┌────┐┌───────────┐┌────┐┌────────────┐
# q: ┤ U3(θ,ϕ,λ) ├  ≡  q: ┤ Rz(λ) ├┤ √X ├┤ Rz(θ + π) ├┤ √X ├┤ Rz(ϕ + 3π) ├
#    └───────────┘        └───────┘└────┘└───────────┘└────┘└────────────┘
q = QuantumRegister(1, "q")
theta = Parameter("theta")
phi = Parameter("phi")
lam = Parameter("lam")
u3_qasm_def = QuantumCircuit(q, global_phase=(lam + phi - pi) / 2)
u3_qasm_def.rz(lam, 0)
u3_qasm_def.sx(0)
u3_qasm_def.rz(theta + pi, 0)
u3_qasm_def.sx(0)
u3_qasm_def.rz(phi + 3 * pi, 0)
_sel.add_equivalence(U3Gate(theta, phi, lam), u3_qasm_def)

# U3Gate
#
#    ┌───────────┐        ┌──────────┐
# q: ┤ U3(θ,ϕ,λ) ├  ≡  q: ┤ U(θ,ϕ,λ) ├
#    └───────────┘        └──────────┘
q = QuantumRegister(1, "q")
theta = Parameter("theta")
phi = Parameter("phi")
lam = Parameter("lam")
u3_to_u = QuantumCircuit(q)
u3_to_u.u(theta, phi, lam, 0)
_sel.add_equivalence(U3Gate(theta, phi, lam), u3_to_u)

# CU3Gate
#                             ┌───────────────┐                                   »
# q_0: ──────■──────     q_0: ┤ U1(λ/2 + ϕ/2) ├──■─────────────────────────────■──»
#      ┌─────┴─────┐  ≡       ├───────────────┤┌─┴─┐┌───────────────────────┐┌─┴─┐»
# q_1: ┤ U3(θ,ϕ,λ) ├     q_1: ┤ U1(λ/2 - ϕ/2) ├┤ X ├┤ U3(-θ/2,0,-λ/2 - ϕ/2) ├┤ X ├»
#      └───────────┘          └───────────────┘└───┘└───────────────────────┘└───┘»
# «
# «q_0: ───────────────
# «     ┌─────────────┐
# «q_1: ┤ U3(θ/2,ϕ,0) ├
# «     └─────────────┘
q = QuantumRegister(2, "q")
theta = Parameter("theta")
phi = Parameter("phi")
lam = Parameter("lam")
def_cu3 = QuantumCircuit(q)
for inst, qargs, cargs in [
    (U1Gate((lam + phi) / 2), [q[0]], []),
    (U1Gate((lam - phi) / 2), [q[1]], []),
    (CXGate(), [q[0], q[1]], []),
    (U3Gate(-theta / 2, 0, -(phi + lam) / 2), [q[1]], []),
    (CXGate(), [q[0], q[1]], []),
    (U3Gate(theta / 2, phi, 0), [q[1]], []),
]:
    def_cu3.append(inst, qargs, cargs)
_sel.add_equivalence(CU3Gate(theta, phi, lam), def_cu3)

q = QuantumRegister(2, "q")
theta = Parameter("theta")
phi = Parameter("phi")
lam = Parameter("lam")
cu3_to_cu = QuantumCircuit(q)
cu3_to_cu.cu(theta, phi, lam, 0, 0, 1)
_sel.add_equivalence(CU3Gate(theta, phi, lam), cu3_to_cu)

# XGate
#
#    ┌───┐        ┌───────────┐
# q: ┤ X ├  ≡  q: ┤ U3(π,0,π) ├
#    └───┘        └───────────┘
q = QuantumRegister(1, "q")
def_x = QuantumCircuit(q)
def_x.append(U3Gate(pi, 0, pi), [q[0]], [])
_sel.add_equivalence(XGate(), def_x)

# XGate
#
#    ┌───┐        ┌───┐┌───┐┌───┐┌───┐
# q: ┤ X ├  ≡  q: ┤ H ├┤ S ├┤ S ├┤ H ├
#    └───┘        └───┘└───┘└───┘└───┘
q = QuantumRegister(1, "q")
def_x = QuantumCircuit(q)
for inst, qargs, cargs in [
    (HGate(), [q[0]], []),
    (SGate(), [q[0]], []),
    (SGate(), [q[0]], []),
    (HGate(), [q[0]], []),
]:
    def_x.append(inst, qargs, cargs)
_sel.add_equivalence(XGate(), def_x)

# XGate
#                 global phase: π/2
#    ┌───┐        ┌───┐┌───┐
# q: ┤ X ├  ≡  q: ┤ Y ├┤ Z ├
#    └───┘        └───┘└───┘
def_x = QuantumCircuit(1, global_phase=pi / 2)
def_x.y(0)
def_x.z(0)
_sel.add_equivalence(XGate(), def_x)

# CXGate

for pos_ry in [False, True]:
    for pos_rxx in [False, True]:
        cx_to_rxx = _cnot_rxx_decompose(pos_ry, pos_rxx)
        _sel.add_equivalence(CXGate(), cx_to_rxx)

# CXGate
#
# q_0: ──■──     q_0: ──────■──────
#      ┌─┴─┐  ≡       ┌───┐ │ ┌───┐
# q_1: ┤ X ├     q_1: ┤ H ├─■─┤ H ├
#      └───┘          └───┘   └───┘
q = QuantumRegister(2, "q")
cx_to_cz = QuantumCircuit(q)
for inst, qargs, cargs in [
    (HGate(), [q[1]], []),
    (CZGate(), [q[0], q[1]], []),
    (HGate(), [q[1]], []),
]:
    cx_to_cz.append(inst, qargs, cargs)
_sel.add_equivalence(CXGate(), cx_to_cz)

# CXGate
#                global phase: 3π/4
#                     ┌───┐     ┌────────┐┌───┐     ┌────────┐┌───┐┌───┐
# q_0: ──■──     q_0: ┤ H ├─────┤0       ├┤ X ├─────┤0       ├┤ H ├┤ S ├─────
#      ┌─┴─┐  ≡       ├───┤┌───┐│  Iswap │├───┤┌───┐│  Iswap │├───┤├───┤┌───┐
# q_1: ┤ X ├     q_1: ┤ X ├┤ H ├┤1       ├┤ X ├┤ H ├┤1       ├┤ S ├┤ X ├┤ H ├
#      └───┘          └───┘└───┘└────────┘└───┘└───┘└────────┘└───┘└───┘└───┘
q = QuantumRegister(2, "q")
cx_to_iswap = QuantumCircuit(q, global_phase=3 * pi / 4)
for inst, qargs, cargs in [
    (HGate(), [q[0]], []),
    (XGate(), [q[1]], []),
    (HGate(), [q[1]], []),
    (iSwapGate(), [q[0], q[1]], []),
    (XGate(), [q[0]], []),
    (XGate(), [q[1]], []),
    (HGate(), [q[1]], []),
    (iSwapGate(), [q[0], q[1]], []),
    (HGate(), [q[0]], []),
    (SGate(), [q[0]], []),
    (SGate(), [q[1]], []),
    (XGate(), [q[1]], []),
    (HGate(), [q[1]], []),
]:
    cx_to_iswap.append(inst, qargs, cargs)
_sel.add_equivalence(CXGate(), cx_to_iswap)

# CXGate
#                global phase: 7π/4
#                     ┌──────────┐┌───────┐┌──────┐
# q_0: ──■──     q_0: ┤ Rz(-π/2) ├┤ Ry(π) ├┤0     ├
#      ┌─┴─┐  ≡       ├─────────┬┘└───────┘│  Ecr │
# q_1: ┤ X ├     q_1: ┤ Rx(π/2) ├──────────┤1     ├
#      └───┘          └─────────┘          └──────┘
q = QuantumRegister(2, "q")
cx_to_ecr = QuantumCircuit(q, global_phase=-pi / 4)
for inst, qargs, cargs in [
    (RZGate(-pi / 2), [q[0]], []),
    (RYGate(pi), [q[0]], []),
    (RXGate(pi / 2), [q[1]], []),
    (ECRGate(), [q[0], q[1]], []),
]:
    cx_to_ecr.append(inst, qargs, cargs)
_sel.add_equivalence(CXGate(), cx_to_ecr)

# CXGate
# q_0: ──■──     q_0: ───────────────■───────────────────
#      ┌─┴─┐  ≡       ┌────────────┐ │P(π) ┌────────────┐
# q_1: ┤ X ├     q_1: ┤ U(π/2,0,π) ├─■─────┤ U(π/2,0,π) ├
#      └───┘          └────────────┘       └────────────┘
q = QuantumRegister(2, "q")
cx_to_cp = QuantumCircuit(q)
for inst, qargs, cargs in [
    (UGate(pi / 2, 0, pi), [q[1]], []),
    (CPhaseGate(pi), [q[0], q[1]], []),
    (UGate(pi / 2, 0, pi), [q[1]], []),
]:
    cx_to_cp.append(inst, qargs, cargs)
_sel.add_equivalence(CXGate(), cx_to_cp)

# CXGate
#                     ┌────────────┐
# q_0: ──■──     q_0: ┤ U(0,0,π/2) ├────■──────────────────
#      ┌─┴─┐  ≡       ├────────────┤┌───┴───┐┌────────────┐
# q_1: ┤ X ├     q_1: ┤ U(π/2,0,π) ├┤ Rz(π) ├┤ U(π/2,0,π) ├
#      └───┘          └────────────┘└───────┘└────────────┘
q = QuantumRegister(2, "q")
cx_to_crz = QuantumCircuit(q)
for inst, qargs, cargs in [
    (UGate(pi / 2, 0, pi), [q[1]], []),
    (UGate(0, 0, pi / 2), [q[0]], []),
    (CRZGate(pi), [q[0], q[1]], []),
    (UGate(pi / 2, 0, pi), [q[1]], []),
]:
    cx_to_crz.append(inst, qargs, cargs)
_sel.add_equivalence(CXGate(), cx_to_crz)

# CXGate
#                global phase: π/4
#                     ┌───────────┐┌─────┐
# q_0: ──■──     q_0: ┤0          ├┤ Sdg ├─
#      ┌─┴─┐  ≡       │  Rzx(π/2) │├─────┴┐
# q_1: ┤ X ├     q_1: ┤1          ├┤ √Xdg ├
#      └───┘          └───────────┘└──────┘
q = QuantumRegister(2, "q")
cx_to_zx90 = QuantumCircuit(q, global_phase=pi / 4)
for inst, qargs, cargs in [
    (RZXGate(pi / 2), [q[0], q[1]], []),
    (SdgGate(), [q[0]], []),
    (SXdgGate(), [q[1]], []),
]:
    cx_to_zx90.append(inst, qargs, cargs)
_sel.add_equivalence(CXGate(), cx_to_zx90)

# CCXGate
#                                                                       ┌───┐
# q_0: ──■──     q_0: ───────────────────■─────────────────────■────■───┤ T ├───■──
#        │                               │             ┌───┐   │  ┌─┴─┐┌┴───┴┐┌─┴─┐
# q_1: ──■──  ≡  q_1: ───────■───────────┼─────────■───┤ T ├───┼──┤ X ├┤ Tdg ├┤ X ├
#      ┌─┴─┐          ┌───┐┌─┴─┐┌─────┐┌─┴─┐┌───┐┌─┴─┐┌┴───┴┐┌─┴─┐├───┤└┬───┬┘└───┘
# q_2: ┤ X ├     q_2: ┤ H ├┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├┤ T ├─┤ H ├──────
#      └───┘          └───┘└───┘└─────┘└───┘└───┘└───┘└─────┘└───┘└───┘ └───┘
q = QuantumRegister(3, "q")
def_ccx = QuantumCircuit(q)
for inst, qargs, cargs in [
    (HGate(), [q[2]], []),
    (CXGate(), [q[1], q[2]], []),
    (TdgGate(), [q[2]], []),
    (CXGate(), [q[0], q[2]], []),
    (TGate(), [q[2]], []),
    (CXGate(), [q[1], q[2]], []),
    (TdgGate(), [q[2]], []),
    (CXGate(), [q[0], q[2]], []),
    (TGate(), [q[1]], []),
    (TGate(), [q[2]], []),
    (HGate(), [q[2]], []),
    (CXGate(), [q[0], q[1]], []),
    (TGate(), [q[0]], []),
    (TdgGate(), [q[1]], []),
    (CXGate(), [q[0], q[1]], []),
]:
    def_ccx.append(inst, qargs, cargs)
_sel.add_equivalence(CCXGate(), def_ccx)

# CCXGate
#
# q_0: ──■──     q_0: ────────■─────────────────■────■───
#        │                  ┌─┴─┐┌─────┐      ┌─┴─┐  │
# q_1: ──■──  ≡  q_1: ──■───┤ X ├┤ Sdg ├──■───┤ X ├──┼───
#      ┌─┴─┐          ┌─┴──┐├───┤└─────┘┌─┴──┐├───┤┌─┴──┐
# q_2: ┤ X ├     q_2: ┤ Sx ├┤ Z ├───────┤ Sx ├┤ Z ├┤ Sx ├
#      └───┘          └────┘└───┘       └────┘└───┘└────┘
q = QuantumRegister(3, "q")
ccx_to_cx_csx = QuantumCircuit(q)
for inst, qargs, cargs in [
    (CSXGate(), [q[1], q[2]], []),
    (CXGate(), [q[0], q[1]], []),
    (ZGate(), [q[2]], []),
    (SdgGate(), [q[1]], []),
    (CSXGate(), [q[1], q[2]], []),
    (ZGate(), [q[2]], []),
    (CXGate(), [q[0], q[1]], []),
    (CSXGate(), [q[0], q[2]], []),
]:
    ccx_to_cx_csx.append(inst, qargs, cargs)
_sel.add_equivalence(CCXGate(), ccx_to_cx_csx)

# YGate
#
#    ┌───┐        ┌───────────────┐
# q: ┤ Y ├  ≡  q: ┤ U3(π,π/2,π/2) ├
#    └───┘        └───────────────┘
q = QuantumRegister(1, "q")
def_y = QuantumCircuit(q)
def_y.append(U3Gate(pi, pi / 2, pi / 2), [q[0]], [])
_sel.add_equivalence(YGate(), def_y)

# YGate
#              global phase: 3π/2
#    ┌───┐        ┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐
# q: ┤ Y ├  ≡  q: ┤ H ├┤ S ├┤ S ├┤ H ├┤ S ├┤ S ├
#    └───┘        └───┘└───┘└───┘└───┘└───┘└───┘
q = QuantumRegister(1, "q")
def_y = QuantumCircuit(q)
def_y.global_phase = 3 * pi / 2
for inst, qargs, cargs in [
    (HGate(), [q[0]], []),
    (SGate(), [q[0]], []),
    (SGate(), [q[0]], []),
    (HGate(), [q[0]], []),
    (SGate(), [q[0]], []),
    (SGate(), [q[0]], []),
]:
    def_y.append(inst, qargs, cargs)
_sel.add_equivalence(YGate(), def_y)

# YGate
#              global phase: π/2
#    ┌───┐        ┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐
# q: ┤ Y ├  ≡  q: ┤ S ├┤ S ├┤ H ├┤ S ├┤ S ├┤ H ├
#    └───┘        └───┘└───┘└───┘└───┘└───┘└───┘
q = QuantumRegister(1, "q")
def_y = QuantumCircuit(q)
def_y.global_phase = pi / 2
for inst, qargs, cargs in [
    (SGate(), [q[0]], []),
    (SGate(), [q[0]], []),
    (HGate(), [q[0]], []),
    (SGate(), [q[0]], []),
    (SGate(), [q[0]], []),
    (HGate(), [q[0]], []),
]:
    def_y.append(inst, qargs, cargs)
_sel.add_equivalence(YGate(), def_y)

# YGate
#                 global phase: π/2
#    ┌───┐        ┌───┐┌───┐
# q: ┤ Y ├  ≡  q: ┤ Z ├┤ X ├
#    └───┘        └───┘└───┘
def_y = QuantumCircuit(1, global_phase=pi / 2)
def_y.z(0)
def_y.x(0)
_sel.add_equivalence(YGate(), def_y)

# CYGate
#
# q_0: ──■──     q_0: ─────────■───────
#      ┌─┴─┐  ≡       ┌─────┐┌─┴─┐┌───┐
# q_1: ┤ Y ├     q_1: ┤ Sdg ├┤ X ├┤ S ├
#      └───┘          └─────┘└───┘└───┘
q = QuantumRegister(2, "q")
def_cy = QuantumCircuit(q)
for inst, qargs, cargs in [
    (SdgGate(), [q[1]], []),
    (CXGate(), [q[0], q[1]], []),
    (SGate(), [q[1]], []),
]:
    def_cy.append(inst, qargs, cargs)
_sel.add_equivalence(CYGate(), def_cy)

# ZGate
#
#    ┌───┐        ┌───────┐
# q: ┤ Z ├  ≡  q: ┤ U1(π) ├
#    └───┘        └───────┘
q = QuantumRegister(1, "q")
def_z = QuantumCircuit(q)
def_z.append(U1Gate(pi), [q[0]], [])
_sel.add_equivalence(ZGate(), def_z)

# ZGate
#
#    ┌───┐        ┌───┐┌───┐
# q: ┤ Z ├  ≡  q: ┤ S ├┤ S ├
#    └───┘        └───┘└───┘
q = QuantumRegister(1, "q")
def_z = QuantumCircuit(q)
for inst, qargs, cargs in [
    (SGate(), [q[0]], []),
    (SGate(), [q[0]], []),
]:
    def_z.append(inst, qargs, cargs)
_sel.add_equivalence(ZGate(), def_z)

# ZGate
#                 global phase: π/2
#    ┌───┐        ┌───┐┌───┐
# q: ┤ Z ├  ≡  q: ┤ X ├┤ Y ├
#    └───┘        └───┘└───┘
def_z = QuantumCircuit(1, global_phase=pi / 2)
def_z.x(0)
def_z.y(0)
_sel.add_equivalence(ZGate(), def_z)

# CZGate
#
# q_0: ─■─     q_0: ───────■───────
#       │   ≡       ┌───┐┌─┴─┐┌───┐
# q_1: ─■─     q_1: ┤ H ├┤ X ├┤ H ├
#                   └───┘└───┘└───┘
q = QuantumRegister(2, "q")
def_cz = QuantumCircuit(q)
for inst, qargs, cargs in [
    (HGate(), [q[1]], []),
    (CXGate(), [q[0], q[1]], []),
    (HGate(), [q[1]], []),
]:
    def_cz.append(inst, qargs, cargs)
_sel.add_equivalence(CZGate(), def_cz)

# CCZGate
#
# q_0: ─■─   q_0: ───────■───────
#       │                │
# q_1: ─■─ = q_1: ───────■───────
#       │         ┌───┐┌─┴─┐┌───┐
# q_2: ─■─   q_2: ┤ H ├┤ X ├┤ H ├
#                 └───┘└───┘└───┘
q = QuantumRegister(3, "q")
def_ccz = QuantumCircuit(q)
for inst, qargs, cargs in [
    (HGate(), [q[2]], []),
    (CCXGate(), [q[0], q[1], q[2]], []),
    (HGate(), [q[2]], []),
]:
    def_ccz.append(inst, qargs, cargs)
_sel.add_equivalence(CCZGate(), def_ccz)

# XGate
#              global phase: π/2
#    ┌───┐        ┌───────┐
# q: ┤ X ├  ≡  q: ┤ Rx(π) ├
#    └───┘        └───────┘
q = QuantumRegister(1, "q")
x_to_rx = QuantumCircuit(q)
x_to_rx.append(RXGate(theta=pi), [q[0]])
x_to_rx.global_phase = pi / 2
_sel.add_equivalence(XGate(), x_to_rx)

# YGate
#              global phase: π/2
#    ┌───┐        ┌───────┐
# q: ┤ Y ├  ≡  q: ┤ Ry(π) ├
#    └───┘        └───────┘
q = QuantumRegister(1, "q")
y_to_ry = QuantumCircuit(q)
y_to_ry.append(RYGate(theta=pi), [q[0]])
y_to_ry.global_phase = pi / 2
_sel.add_equivalence(YGate(), y_to_ry)

# HGate
#              global phase: π/2
#    ┌───┐        ┌─────────┐┌───────┐
# q: ┤ H ├  ≡  q: ┤ Ry(π/2) ├┤ Rx(π) ├
#    └───┘        └─────────┘└───────┘
q = QuantumRegister(1, "q")
h_to_rxry = QuantumCircuit(q)
h_to_rxry.append(RYGate(theta=pi / 2), [q[0]])
h_to_rxry.append(RXGate(theta=pi), [q[0]])
h_to_rxry.global_phase = pi / 2
_sel.add_equivalence(HGate(), h_to_rxry)

# HGate
#              global phase: π/2
#    ┌───┐        ┌────────────┐┌────────┐
# q: ┤ H ├  ≡  q: ┤ R(π/2,π/2) ├┤ R(π,0) ├
#    └───┘        └────────────┘└────────┘
q = QuantumRegister(1, "q")
h_to_rr = QuantumCircuit(q)
h_to_rr.append(RGate(theta=pi / 2, phi=pi / 2), [q[0]])
h_to_rr.append(RGate(theta=pi, phi=0), [q[0]])
h_to_rr.global_phase = pi / 2
_sel.add_equivalence(HGate(), h_to_rr)

# XXPlusYYGate
# ┌───────────────┐
# ┤0              ├
# │  {XX+YY}(θ,β) │
# ┤1              ├
# └───────────────┘
#    ┌───────┐  ┌───┐            ┌───┐┌────────────┐┌───┐  ┌─────┐   ┌────────────┐
#   ─┤ Rz(β) ├──┤ S ├────────────┤ X ├┤ Ry(-0.5*θ) ├┤ X ├──┤ Sdg ├───┤ Rz(-1.0*β) ├───────────
# ≡ ┌┴───────┴─┐├───┴┐┌─────────┐└─┬─┘├────────────┤└─┬─┘┌─┴─────┴──┐└──┬──────┬──┘┌─────────┐
#   ┤ Rz(-π/2) ├┤ √X ├┤ Rz(π/2) ├──■──┤ Ry(-0.5*θ) ├──■──┤ Rz(-π/2) ├───┤ √Xdg ├───┤ Rz(π/2) ├
#   └──────────┘└────┘└─────────┘     └────────────┘     └──────────┘   └──────┘   └─────────┘
q = QuantumRegister(2, "q")
xxplusyy = QuantumCircuit(q)
beta = Parameter("beta")
theta = Parameter("theta")
rules: list[tuple[Gate, list[Qubit], list[Clbit]]] = [
    (RZGate(beta), [q[0]], []),
    (RZGate(-pi / 2), [q[1]], []),
    (SXGate(), [q[1]], []),
    (RZGate(pi / 2), [q[1]], []),
    (SGate(), [q[0]], []),
    (CXGate(), [q[1], q[0]], []),
    (RYGate(-theta / 2), [q[1]], []),
    (RYGate(-theta / 2), [q[0]], []),
    (CXGate(), [q[1], q[0]], []),
    (SdgGate(), [q[0]], []),
    (RZGate(-pi / 2), [q[1]], []),
    (SXdgGate(), [q[1]], []),
    (RZGate(pi / 2), [q[1]], []),
    (RZGate(-beta), [q[0]], []),
]
for instr, qargs, cargs in rules:
    xxplusyy._append(instr, qargs, cargs)
_sel.add_equivalence(XXPlusYYGate(theta, beta), xxplusyy)

# XXMinusYYGate
# ┌───────────────┐
# ┤0              ├
# │  {XX-YY}(θ,β) │
# ┤1              ├
# └───────────────┘
#    ┌──────────┐ ┌────┐┌─────────┐      ┌─────────┐       ┌──────────┐ ┌──────┐┌─────────┐
#   ─┤ Rz(-π/2) ├─┤ √X ├┤ Rz(π/2) ├──■───┤ Ry(θ/2) ├────■──┤ Rz(-π/2) ├─┤ √Xdg ├┤ Rz(π/2) ├
# ≡ ┌┴──────────┴┐├───┬┘└─────────┘┌─┴─┐┌┴─────────┴─┐┌─┴─┐└─┬─────┬──┘┌┴──────┤└─────────┘
#   ┤ Rz(-1.0*β) ├┤ S ├────────────┤ X ├┤ Ry(-0.5*θ) ├┤ X ├──┤ Sdg ├───┤ Rz(β) ├───────────
#   └────────────┘└───┘            └───┘└────────────┘└───┘  └─────┘   └───────┘
q = QuantumRegister(2, "q")
xxminusyy = QuantumCircuit(q)
beta = Parameter("beta")
theta = Parameter("theta")
rules: list[tuple[Gate, list[Qubit], list[Clbit]]] = [
    (RZGate(-beta), [q[1]], []),
    (RZGate(-pi / 2), [q[0]], []),
    (SXGate(), [q[0]], []),
    (RZGate(pi / 2), [q[0]], []),
    (SGate(), [q[1]], []),
    (CXGate(), [q[0], q[1]], []),
    (RYGate(theta / 2), [q[0]], []),
    (RYGate(-theta / 2), [q[1]], []),
    (CXGate(), [q[0], q[1]], []),
    (SdgGate(), [q[1]], []),
    (RZGate(-pi / 2), [q[0]], []),
    (SXdgGate(), [q[0]], []),
    (RZGate(pi / 2), [q[0]], []),
    (RZGate(beta), [q[1]], []),
]
for instr, qargs, cargs in rules:
    xxminusyy._append(instr, qargs, cargs)
_sel.add_equivalence(XXMinusYYGate(theta, beta), xxminusyy)
