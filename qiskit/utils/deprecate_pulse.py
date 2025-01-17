# This code is part of Qiskit.
#
# (C) Copyright IBM 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Deprecation functions for Qiskit Pulse. To be removed in Qiskit 2.0.
"""

import warnings
import functools

from qiskit.utils.deprecation import deprecate_func, deprecate_arg


def deprecate_pulse_func(func):
    """Deprecation message for functions and classes"""
    return deprecate_func(
        since="1.3",
        package_name="Qiskit",
        removal_timeline="in Qiskit 2.0",
        additional_msg="The entire Qiskit Pulse package is being deprecated "
        "and will be moved to the Qiskit Dynamics repository: "
        "https://github.com/qiskit-community/qiskit-dynamics",
    )(func)


def deprecate_pulse_dependency(*args, moving_to_dynamics: bool = False, **kwargs):
    # pylint: disable=missing-param-doc
    """Deprecation message for functions and classes which use or depend on Pulse

    Args:
        moving_to_dynamics: set to True if the dependency is moving to Qiskit Dynamics. This affects
            the deprecation message being printed, namely saying explicitly whether the dependency will
            be moved to Qiskit Dynamics or whether it will just be removed without an alternative.
    """

    def msg_handler(func):
        fully_qual_name = format(f"{func.__module__}.{func.__qualname__}")
        if ".__init__" in fully_qual_name:  # Deprecating a class' vis it __init__ method
            fully_qual_name = fully_qual_name[:-9]
        elif "is_property" not in kwargs:  # Deprecating either a function or a method
            fully_qual_name += "()"

        message = (
            "The entire Qiskit Pulse package is being deprecated and will be moved to the Qiskit "
            "Dynamics repository: https://github.com/qiskit-community/qiskit-dynamics."
            + (
                format(f" Note that ``{fully_qual_name}`` will be moved as well.")
                if moving_to_dynamics
                else format(
                    f" Note that once removed, ``{fully_qual_name}`` will have no alternative in Qiskit."
                )
            )
        )

        decorator = deprecate_func(
            since="1.3",
            package_name="Qiskit",
            removal_timeline="in Qiskit 2.0",
            additional_msg=message,
            **kwargs,
        )(func)

        # Taken when `deprecate_pulse_dependency` is used with no arguments and with empty parentheses,
        # in which case the decorated function is passed

        return decorator

    if args:
        return msg_handler(args[0])
    return msg_handler


def deprecate_pulse_arg(arg_name: str, **kwargs):
    """Deprecation message for arguments related to Pulse"""
    return deprecate_arg(
        name=arg_name,
        since="1.3",
        package_name="Qiskit",
        removal_timeline="in Qiskit 2.0",
        additional_msg="The entire Qiskit Pulse package is being deprecated "
        "and this argument uses a dependency on the package.",
        **kwargs,
    )


def ignore_pulse_deprecation_warnings(func):
    """Ignore deprecation warnings emitted from the pulse package"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=DeprecationWarning, message="The (.*) ``qiskit.pulse"
            )
            return func(*args, **kwargs)

    return wrapper


def decorate_test_methods(decorator):
    """Put a given decorator on all the decorated class methods whose name starts with `test_`"""

    def cls_wrapper(cls):
        for attr in dir(cls):
            if attr.startswith("test_") and callable(object.__getattribute__(cls, attr)):
                setattr(cls, attr, decorator(object.__getattribute__(cls, attr)))

        return cls

    return cls_wrapper
