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

"""Base TestCases for the unit tests.

Implementors of unit tests for Qiskit should subclass
``QiskitTestCase`` in order to take advantage of utility functions (for example,
the environment variables for customizing different options), and the
decorators in the ``decorators`` package.
"""

import inspect
import logging
import os
import sys
import warnings
import unittest
from unittest.util import safe_repr

from qiskit.utils.parallel import get_platform_parallel_default
from qiskit.exceptions import QiskitWarning
from qiskit.utils import optionals as _optionals
from qiskit.circuit import QuantumCircuit
from .decorators import enforce_subclasses_call


__unittest = True  # Allows shorter stack trace for .assertDictAlmostEqual


# If testtools is installed use that as a (mostly) drop in replacement for
# unittest's TestCase. This will enable the fixtures used for capturing stdout
# stderr, and pylogging to attach the output to stestr's result stream.
if _optionals.HAS_TESTTOOLS:
    import testtools

    class BaseTestCase(testtools.TestCase):
        """Base test class."""

        # testtools maintains their own version of assert functions which mostly
        # behave as value adds to the std unittest assertion methods. However,
        # for assertEquals and assertRaises modern unittest has diverged from
        # the forks in testtools and offer more (or different) options that are
        # incompatible testtools versions. Just use the stdlib versions so that
        # our tests work as expected.
        assertRaises = unittest.TestCase.assertRaises
        assertEqual = unittest.TestCase.assertEqual

else:

    class BaseTestCase(unittest.TestCase):
        """Base test class."""

        pass


@enforce_subclasses_call(["setUp", "setUpClass", "tearDown", "tearDownClass"])
class QiskitTestCase(BaseTestCase):
    """Additions for Qiskit test cases."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Set logging to file and stdout if the LOG_LEVEL envar is set.
        cls.log = logging.getLogger(cls.__name__)

        if log_level := os.getenv("LOG_LEVEL"):
            log_fmt = f"{cls.log.name}.%(funcName)s:%(levelname)s:%(asctime)s: %(message)s"
            formatter = logging.Formatter(log_fmt)
            file_handler = logging.FileHandler(f"{os.path.splitext(inspect.getfile(cls))[0]}.log")
            file_handler.setFormatter(formatter)
            cls.log.addHandler(file_handler)

            if os.getenv("STREAM_LOG"):
                # Set up the stream handler.
                stream_handler = logging.StreamHandler()
                stream_handler.setFormatter(formatter)
                cls.log.addHandler(stream_handler)

            # Set the logging level from the environment variable, defaulting
            # to INFO if it is not a valid level.
            level = logging._nameToLevel.get(log_level, logging.INFO)
            cls.log.setLevel(level)

        warnings.filterwarnings("error", category=DeprecationWarning)
        warnings.filterwarnings("error", category=QiskitWarning)

        # Numpy 2 made a few new modules private, and have warnings that trigger if you try to
        # access attributes that _would_ have existed.  Unfortunately, Python's `warnings` module
        # adds a field called `__warningregistry__` to any module that triggers a warning, and
        # `unittest.TestCase.assertWarns` then queries said fields on all existing modules.  On
        # macOS ARM, we see some (we think harmless) warnings come out of `numpy.linalg._linalg` (a
        # now-private module) during transpilation, which means that subsequent `assertWarns` calls
        # can spuriously trick Numpy into sending out a nonsense `DeprecationWarning`.
        # Tracking issue: https://github.com/Qiskit/qiskit/issues/12679
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=r".*numpy\.(\w+\.)*__warningregistry__",
        )

        # We only use pandas transitively through seaborn, so it's their responsibility to mark if
        # their use of pandas would be a problem.
        warnings.filterwarnings(
            "default",
            category=DeprecationWarning,
            # The `(?s)` magic is to force use of the `re.DOTALL` flag, because the Pandas message
            # includes hard-break newlines all over the place.
            message="(?s).*Pyarrow.*required dependency.*next major release of pandas",
            module=r"seaborn(\..*)?",
        )

        # Safe to remove once https://github.com/Qiskit/qiskit-aer/pull/2179 is in a release version
        # of Aer.
        warnings.filterwarnings(
            "ignore",  # If "default", it floods the CI output
            category=DeprecationWarning,
            message="Treating CircuitInstruction as an iterable is deprecated",
            module=r"qiskit_aer(\.[a-zA-Z0-9_]+)*",
        )

        # Safe to remove once https://github.com/Qiskit/qiskit-aer/issues/2197 is in a release version
        # of Aer.
        warnings.filterwarnings(
            "ignore",  # If "default", it floods the CI output
            category=DeprecationWarning,
            message=r".*qiskit\.providers\.models.*",
            module=r"qiskit_aer(\.[a-zA-Z0-9_]+)*",
        )

        # Safe to remove once https://github.com/Qiskit/qiskit-aer/issues/2065 is in a release version
        # of Aer.
        warnings.filterwarnings(
            "ignore",  # If "default", it floods the CI output
            category=DeprecationWarning,
            message=r".*The `Qobj` class and related functionality.*",
            module=r"qiskit_aer",
        )

        # Safe to remove once https://github.com/Qiskit/qiskit-aer/pull/2184 is in a release version
        # of Aer.
        warnings.filterwarnings(
            "ignore",  # If "default", it floods the CI output
            category=DeprecationWarning,
            message=r".*The abstract Provider and ProviderV1 classes are deprecated.*",
            module="qiskit_aer",
        )

        # Safe to remove once `FakeBackend` is removed (2.0)
        warnings.filterwarnings(
            "ignore",  # If "default", it floods the CI output
            category=DeprecationWarning,
            message=r".*from_backend using V1 based backend is deprecated as of Aer 0.15*",
            module="qiskit.providers.fake_provider.fake_backend",
        )

        allow_DeprecationWarning_message = [
            r"The property ``qiskit\.circuit\.bit\.Bit\.(register|index)`` is deprecated.*",
        ]
        for msg in allow_DeprecationWarning_message:
            warnings.filterwarnings("default", category=DeprecationWarning, message=msg)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__setup_called = False
        self.__teardown_called = False

    def setUp(self):
        super().setUp()
        self.addTypeEqualityFunc(QuantumCircuit, self.assertQuantumCircuitEqual)
        if self.__setup_called:
            raise ValueError(
                f"In File: {(sys.modules[self.__class__.__module__].__file__,)}\n"
                "TestCase.setUp was already called. Do not explicitly call "
                "setUp from your tests. In your own setUp, use super to call "
                "the base setUp."
            )
        self.__setup_called = True

    def tearDown(self):
        super().tearDown()
        if self.__teardown_called:
            raise ValueError(
                f"In File: {(sys.modules[self.__class__.__module__].__file__,)}\n"
                "TestCase.tearDown was already called. Do not explicitly call "
                "tearDown from your tests. In your own tearDown, use super to "
                "call the base tearDown."
            )
        self.__teardown_called = True

        # Reset the default providers, as in practice they acts as a singleton
        # due to importing the instances from the top-level qiskit namespace.
        from qiskit.providers.basic_provider import BasicProvider

        with self.assertWarns(DeprecationWarning):
            BasicProvider()._backends = BasicProvider()._verify_backends()

    def assertQuantumCircuitEqual(self, qc1, qc2, msg=None):
        """Extra assertion method to give a better error message when two circuits are unequal."""
        if qc1 == qc2:
            return
        if msg is None:
            msg = "The two circuits are not equal."
        msg += f"""
Left circuit:
{qc1}

Right circuit:
{qc2}"""
        raise self.failureException(msg)

    def assertDictAlmostEqual(
        self, dict1, dict2, delta=None, msg=None, places=None, default_value=0
    ):
        """Assert two dictionaries with numeric values are almost equal.

        Fail if the two dictionaries are unequal as determined by
        comparing that the difference between values with the same key are
        not greater than delta (default 1e-8), or that difference rounded
        to the given number of decimal places is not zero. If a key in one
        dictionary is not in the other the default_value keyword argument
        will be used for the missing value (default 0). If the two objects
        compare equal then they will automatically compare almost equal.

        Args:
            dict1 (dict): a dictionary.
            dict2 (dict): a dictionary.
            delta (number): threshold for comparison (defaults to 1e-8).
            msg (str): return a custom message on failure.
            places (int): number of decimal places for comparison.
            default_value (number): default value for missing keys.

        Raises:
            TypeError: if the arguments are not valid (both `delta` and
                `places` are specified).
            AssertionError: if the dictionaries are not almost equal.
        """

        error_msg = dicts_almost_equal(dict1, dict2, delta, places, default_value)

        if error_msg:
            msg = self._formatMessage(msg, error_msg)
            raise self.failureException(msg)

    def enable_parallel_processing(self):
        """
        Enables parallel processing, for the duration of a test, on platforms
        that support it. This is done by temporarily overriding the value of
        the QISKIT_PARALLEL environment variable with the platform specific default.
        """
        parallel_default = str(get_platform_parallel_default()).upper()

        def set_parallel_env(name, value):
            os.environ[name] = value

        self.addCleanup(
            lambda value: set_parallel_env("QISKIT_PARALLEL", value),
            os.getenv("QISKIT_PARALLEL", parallel_default),
        )

        os.environ["QISKIT_PARALLEL"] = parallel_default


class FullQiskitTestCase(QiskitTestCase):
    """further additions for Qiskit test cases, if ``testtools`` is available.

    It is not normally safe to derive from this class by name; on import, Qiskit checks if the
    necessary packages are available, and binds this class to the name :obj:`~QiskitTestCase` if so.
    If you derive directly from it, you may try and instantiate the class without satisfying its
    dependencies."""

    @_optionals.HAS_FIXTURES.require_in_call("output-capturing test cases")
    def setUp(self):
        import fixtures

        super().setUp()
        if os.environ.get("QISKIT_TEST_CAPTURE_STREAMS"):
            stdout = self.useFixture(fixtures.StringStream("stdout")).stream
            self.useFixture(fixtures.MonkeyPatch("sys.stdout", stdout))
            stderr = self.useFixture(fixtures.StringStream("stderr")).stream
            self.useFixture(fixtures.MonkeyPatch("sys.stderr", stderr))
            self.useFixture(fixtures.LoggerFixture(nuke_handlers=False, level=None))


def dicts_almost_equal(dict1, dict2, delta=None, places=None, default_value=0):
    """Test if two dictionaries with numeric values are almost equal.

    Fail if the two dictionaries are unequal as determined by
    comparing that the difference between values with the same key are
    not greater than delta (default 1e-8), or that difference rounded
    to the given number of decimal places is not zero. If a key in one
    dictionary is not in the other the default_value keyword argument
    will be used for the missing value (default 0). If the two objects
    compare equal then they will automatically compare almost equal.

    Args:
        dict1 (dict): a dictionary.
        dict2 (dict): a dictionary.
        delta (number): threshold for comparison (defaults to 1e-8).
        places (int): number of decimal places for comparison.
        default_value (number): default value for missing keys.

    Raises:
        TypeError: if the arguments are not valid (both `delta` and
            `places` are specified).

    Returns:
        String: Empty string if dictionaries are almost equal. A description
            of their difference if they are deemed not almost equal.
    """

    def valid_comparison(value):
        """compare value to delta, within places accuracy"""
        if places is not None:
            return round(value, places) == 0
        else:
            return value < delta

    # Check arguments.
    if dict1 == dict2:
        return ""
    if places is not None:
        if delta is not None:
            raise TypeError("specify delta or places not both")
        msg_suffix = f" within {places} places"
    else:
        delta = delta or 1e-8
        msg_suffix = f" within {delta} delta"

    # Compare all keys in both dicts, populating error_msg.
    error_msg = ""
    for key in set(dict1.keys()) | set(dict2.keys()):
        val1 = dict1.get(key, default_value)
        val2 = dict2.get(key, default_value)
        if not valid_comparison(abs(val1 - val2)):
            error_msg += f"({safe_repr(key)}: {safe_repr(val1)} != {safe_repr(val2)}), "

    if error_msg:
        return error_msg[:-2] + msg_suffix
    else:
        return ""


if _optionals.HAS_TESTTOOLS and _optionals.HAS_FIXTURES:
    QiskitTestCase = FullQiskitTestCase
