# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=too-many-return-statements

"""Check if number close to values of PI
"""

import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.exceptions import QiskitError

MAX_FRAC = 16
N, D = np.meshgrid(np.arange(1, MAX_FRAC + 1), np.arange(1, MAX_FRAC + 1))
FRAC_MESH = N / D * np.pi
RECIP_MESH = N / D / np.pi
POW_LIST = np.pi ** np.arange(2, 5)


def pi_check(inpt, eps=1e-9, output="text", ndigits=None):
    """Computes if a number is close to an integer
    fraction or multiple of PI and returns the
    corresponding string.

    Args:
        inpt (float): Number to check.
        eps (float): EPS to check against.
        output (str): Options are 'text' (default),
                      'latex', 'mpl', and 'qasm'.
        ndigits (int or None): Number of digits to print
                               if returning raw inpt.
                               If `None` (default), Python's
                               default float formatting is used.

    Returns:
        str: string representation of output.

    Raises:
        QiskitError: if output is not a valid option.
    """
    if isinstance(inpt, ParameterExpression):
        param_str = str(inpt)
        from sympy import sympify

        expr = sympify(inpt._symbol_expr)
        syms = expr.atoms()
        for sym in syms:
            if not sym.is_number:
                continue
            pi = pi_check(abs(float(sym)), eps=eps, output=output, ndigits=ndigits)
            try:
                _ = float(pi)
            except (ValueError, TypeError):
                from sympy import sstr

                sym_str = sstr(abs(sym), full_prec=False)
                param_str = param_str.replace(sym_str, pi)
        return param_str
    elif isinstance(inpt, str):
        return inpt

    def normalize(single_inpt):
        if abs(single_inpt) < eps:
            return "0"

        if output == "text":
            pi = "π"
        elif output == "qasm":
            pi = "pi"
        elif output == "latex":
            pi = "\\pi"
        elif output == "mpl":
            pi = "$\\pi$"
        else:
            raise QiskitError("pi_check parameter output should be text, latex, mpl, or qasm.")

        neg_str = "-" if single_inpt < 0 else ""

        # First check is for whole multiples of pi
        val = single_inpt / np.pi
        if abs(val) >= 1 - eps:
            if abs(abs(val) - abs(round(val))) < eps:
                val = int(abs(round(val)))
                if abs(val) == 1:
                    str_out = f"{neg_str}{pi}"
                else:
                    if output == "qasm":
                        str_out = f"{neg_str}{val}*{pi}"
                    else:
                        str_out = f"{neg_str}{val}{pi}"
                return str_out

        # Second is a check for powers of pi
        if abs(single_inpt) > np.pi:
            power = np.where(abs(abs(single_inpt) - POW_LIST) < eps)
            if power[0].shape[0]:
                if output == "qasm":
                    if ndigits is None:
                        str_out = f"{single_inpt}"
                    else:
                        str_out = "{:.{}g}".format(single_inpt, ndigits)
                elif output == "latex":
                    str_out = f"{neg_str}{pi}^{power[0][0] + 2}"
                elif output == "mpl":
                    str_out = f"{neg_str}{pi}$^{power[0][0] + 2}$"
                else:
                    str_out = f"{neg_str}{pi}**{power[0][0] + 2}"
                return str_out

        # Third is a check for a number larger than MAX_FRAC * pi, not a
        # multiple or power of pi, since no fractions will exceed MAX_FRAC * pi
        if abs(single_inpt) >= (MAX_FRAC * np.pi):
            if ndigits is None:
                str_out = f"{single_inpt}"
            else:
                str_out = "{:.{}g}".format(single_inpt, ndigits)
            return str_out

        # Fourth check is for fractions for 1*pi in the numer and any
        # number in the denom.
        val = np.pi / single_inpt
        if abs(abs(val) - abs(round(val))) < eps:
            val = int(abs(round(val)))
            if output == "latex":
                str_out = f"\\frac{{{neg_str}{pi}}}{{{val}}}"
            else:
                str_out = f"{neg_str}{pi}/{val}"
            return str_out

        # Fifth check is for fractions where the numer > 1*pi and numer
        # is up to MAX_FRAC*pi and denom is up to MAX_FRAC and all
        # fractions are reduced. Ex. 15pi/16, 2pi/5, 15pi/2, 16pi/9.
        frac = np.where(np.abs(abs(single_inpt) - FRAC_MESH) < eps)
        if frac[0].shape[0]:
            numer = int(frac[1][0]) + 1
            denom = int(frac[0][0]) + 1
            if output == "latex":
                str_out = f"\\frac{{{neg_str}{numer}{pi}}}{{{denom}}}"
            elif output == "qasm":
                str_out = f"{neg_str}{numer}*{pi}/{denom}"
            else:
                str_out = f"{neg_str}{numer}{pi}/{denom}"
            return str_out

        # Sixth check is for fractions where the numer > 1 and numer
        # is up to MAX_FRAC and denom is up to MAX_FRAC*pi and all
        # fractions are reduced. Ex. 15/16pi, 2/5pi, 15/2pi, 16/9pi
        frac = np.where(np.abs(abs(single_inpt) - RECIP_MESH) < eps)
        if frac[0].shape[0]:
            numer = int(frac[1][0]) + 1
            denom = int(frac[0][0]) + 1
            if denom == 1 and output != "qasm":
                denom = ""
            if output == "latex":
                str_out = f"\\frac{{{neg_str}{numer}}}{{{denom}{pi}}}"
            elif output == "qasm":
                str_out = f"{neg_str}{numer}/({denom}*{pi})"
            else:
                str_out = f"{neg_str}{numer}/{denom}{pi}"
            return str_out

        # Nothing found
        if ndigits is None:
            str_out = f"{single_inpt}"
        else:
            str_out = "{:.{}g}".format(single_inpt, ndigits)
        return str_out

    complex_inpt = complex(inpt)
    real, imag = map(normalize, [complex_inpt.real, complex_inpt.imag])

    jstr = "\\jmath" if output == "latex" else "j"
    if real == "0" and imag != "0":
        str_out = imag + jstr
    elif real != "0" and imag != "0":
        op_str = "+"
        # Remove + if imag negative except for latex fractions
        if complex_inpt.imag < 0 and (output != "latex" or "\\frac" not in imag):
            op_str = ""
        str_out = f"{real}{op_str}{imag}{jstr}"
    else:
        str_out = real
    return str_out
